import torch
#import resnet
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time, logging
#import redisai
#import redis
import heapdict
import os
import sys
import argparse
import warnings
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchtoolbox import metric
from importlib import import_module
from torch.utils.data.shadedataset import ShadeDataset, ShadeValDataset


warnings.filterwarnings("ignore")
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
#logger.addHandler(filehandler)
logger.addHandler(streamhandler)

PQ = heapdict.heapdict()
ghost_cache = heapdict.heapdict()
id_size_map = dict()

num_training_samples = 1281167

def train(args):
    print("begin trainning...")
    global PQ
    global ghost_cache
    global id_size_map
	
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs
    BATCH_SIZE = args.batch_size
    workers = args.workers
    cache_size = args.working_set_size
    batches_pre_epoch = num_training_samples // BATCH_SIZE

    model = import_module('res_model').resnet('resnet50')

    model.to(device)

    
    #loss_func = nn.NLLLoss()
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,  momentum=0.9, weight_decay = 1e-4, nesterov=True)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    top1_acc = metric.Accuracy(name='Top1 Accuracy')
    top5_acc = metric.TopKAccuracy(top=5, name='Top5 Accuracy')
    loss_record = metric.NumericalCost(name='Loss')

    # ============================ TEST 0 original prep========================================
     
    transform_train = transforms.Compose([
                transforms.Resize(size=256),
        	transforms.CenterCrop(size=224),
                transforms.RandomHorizontalFlip(),
        	transforms.ToTensor(),
        	transforms.Normalize([0.485, 0.456, 0.406],
                             	     [0.229, 0.224, 0.225])
	    ])
    
    transform_valid = transforms.Compose([
		transforms.Resize(size=256),
        	transforms.CenterCrop(size=224),
        	transforms.ToTensor(),
        	transforms.Normalize([0.485, 0.456, 0.406],
                             	     [0.229, 0.224, 0.225])
	   ])
    # ============================ TEST 1 cache tensor========================================
    """
    transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.CenterCrop(size=224),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=1)
            ])
    """
    # =========================== TEST 2 until resize======================================= 
    """
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    """
    # =========================== TEST 3 until normalize===================================
   
    """    
    transform_train = transforms.Compose([
            transforms.CenterCrop(size=224),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=1)
            ])
    """

    # =========================== TEST 4 until CenterCrop===================================
    """ 
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=1)
            ])
    """
    # =============================== Redis ===============================================    
    
    train_directory = args.train_paths
    train_imagefolder = datasets.ImageFolder(train_directory[0])
    train_imagefolder_list = []
    train_imagefolder_list.append(train_imagefolder)

    train_dataset = ShadeDataset(
    		        train_imagefolder_list,
    		        transform_train,
    		        PQ=PQ, ghost_cache=ghost_cache, size_map=id_size_map, wss=cache_size, worker_nums=workers, key_counter=1)
    
    # ======================= TEST 5 non-cache ===========================================
    dataset = '/data/hanke/dataset'
    #train_directory = os.path.join(dataset, 'train')
    test_directory = os.path.join(dataset, 'val')
    #train_dataset = datasets.ImageFolder(root=train_directory, transform=transform_train)
    #--------------------------------------------------------------------------------------
    test_dataset  = datasets.ImageFolder(root=test_directory, transform=transform_valid)
    train_data_size = len(train_dataset)
    valid_data_size = len(test_dataset)
      
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                    batch_size=BATCH_SIZE,
		    shuffle=True,
		    num_workers=workers)
		    #pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                     batch_size=BATCH_SIZE,
                   shuffle=True,
                   num_workers=workers)


    print("dataset size is %d" % len(train_dataset))

    #---------------------------------------------------
    total_time = 0
    history = []
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(epochs):
        #if epoch == 0:
        #   train_dataset.set_current_cache_size(1281167)
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        top1_acc.reset()
        loss_record.reset()
        tic = time.time()

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

	######
        sum_loss = 0.0
        correct = 0.0
        total = 0.0


        # =================== test for mem fix =============================
        for i, (inputs, labels, indices) in enumerate(tqdm(train_loader)):
        #
        # ==================  test for disk read ==========================
        #for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            top1_acc.update(outputs, labels)
            loss_record.update(loss)

            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
       
        #train_dataset.sort_by_imagesize()
        #train_dataset.set_current_cache_size(1281167)
        #sys.exit()
        train_speed = int(num_training_samples // (time.time() - tic))
        epoch_msg = 'Train Epoch {}: {}:{:.5}, {}:{:.5}, {} samples/s.'.format(
            epoch, top1_acc.name, top1_acc.get(), loss_record.name, loss_record.get(), train_speed)
        logger.info(epoch_msg)
        # valid
        with torch.no_grad():
            top1_acc.reset()
            top5_acc.reset()
            loss_record.reset()
            model.eval()
            for j, (inputs, labels) in enumerate(tqdm(test_loader)):    
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                # top1 & top5
                top1_acc.update(outputs, labels)
                top5_acc.update(outputs, labels)
                loss_record.update(loss)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        test_msg = 'Test Epoch {}: {}:{:.5}, {}:{:.5}, {}:{:.5}\n'.format(epoch, top1_acc.name, top1_acc.get(), top5_acc.name, top5_acc.get(),
                    loss_record.name, loss_record.get())
        logger.info(test_msg)

        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size

        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()
        
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_valid_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        scheduler.step()
        #torch.save(model, 'models/'+dataset+'_model_'+str(epoch+1)+'.pt')
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_paths', nargs="+", help="Paths to the train set")
    parser.add_argument('-n', '--workers', default=0, type=int, metavar='N',
						help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
						help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size used for training and validation")
    parser.add_argument('-wss','--working_set_size', default=0.1, type=float, help='percentage of dataset to be cached.')
    args = parser.parse_args()
    for i in range(len(args.train_paths)):
        args.train_paths[i] = os.path.abspath(args.train_paths[i])

    for i in range(len(args.train_paths)):
        print('train_path %d %s' %(i,args.train_paths[i]))

    print('worker_numbers: %d' % (args.workers))
    print('epochs: %d' % (args.epochs))
    print('batch_size: %d' % (args.batch_size))
    print('wss: %f' % (args.working_set_size))

    train(args)

main()
