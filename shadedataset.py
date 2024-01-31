import bisect
import random
import warnings

from torch._utils import _accumulate
from torch import randperm
# No 'default_generator' in torch/__init__.pyi
from torch import default_generator  # type: ignore
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple
from ... import Tensor, Generator

import math

import torch
from . import Sampler, Dataset
import torch.distributed as dist

import os
import time
from datetime import datetime
import argparse
#import torchvision
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import random
#import pandas
import PIL.Image as Image
import numpy as np
#import redis
import io
from io import BytesIO
import numpy as np
from torch._utils import ExceptionWrapper
import redisai
import redis
import heapdict
import PIL
from rediscluster import RedisCluster
from collections import OrderedDict
import torchvision.transforms as transforms
import sys
import base64


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

#class ShadeDataset(torch.utils.data.Dataset):
class ShadeDataset(Dataset):

	def __init__(self, imagefolders, transform=None, target_transform=None, cache_data = False,
		PQ=None, ghost_cache=None, key_counter= None, size_map=None,
		wss = 0.1, worker_nums=1, host_ip = '0.0.0.0', port_num = '6379'):
		_datasets = []
		self.samples = []
		self.classes = []
		self.transform = transform
		self.target_transform = target_transform
		self.loader = None
		self.cache_data = cache_data
		self.workers = worker_nums
		self.wss = wss
		for imagefolder in imagefolders:
			#dataset = torchvision.datasets.ImageFolder(root)
			dataset = imagefolder
			self.loader = dataset.loader
			_datasets.append(dataset)
			self.samples.extend(dataset.samples)
			self.classes.extend(dataset.classes)
		self.classes = list(set(self.classes))

		self.cache_portion = self.wss * len(self.samples)
		self.cache_portion = int(self.cache_portion // (self.workers))

		self.id_tensor_map = dict()
		self.load_map = False

		if host_ip == '0.0.0.0':
			#self.key_id_map = redis.Redis()
			self.key_id_map = redisai.Client(host='localhost', port=6379)
			self.key_id_map_o = redis.Redis(host='localhost', port=3306)
		else:
			self.startup_nodes = [{"host": host_ip, "port": port_num}]
			self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)

		self.PQ = PQ
		self.ghost_cache = ghost_cache
		self.key_counter = key_counter
		self.cur_cache_size = 0
		
	def random_func(self):
		return 0.6858089651836363

	def set_num_local_samples(self,n):
		self.key_counter = n

	def set_PQ(self,curr_PQ):
		self.PQ = curr_PQ

	def set_ghost_cache(self,curr_ghost_cache):
		self.ghost_cache = curr_ghost_cache

	def get_PQ(self):
		return self.PQ

	def get_ghost_cache(self):
		return self.ghost_cache

	def get_cache_portion(self):
		print(self.cache_portion)

	"""
	def transform_to_tensor(self, sample):
		tensor = transforms.ToTensor()(sample)
		resize_img = transforms.Resize((128, 128))(tensor)
		norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(resize_img)
		return transforms.CenterCrop(size=224)(norm)
		#return transforms.ToTensor()(sample)
		#return transforms.Resize((128, 128))(tensor)
	"""
	def set_current_cache_size(self, n):
		self.cur_cache_size = n

	def transform_to_tensor(self, sample):
		tensor = transforms.ToTensor()(sample)
		return transforms.Resize(size=256)(tensor)
		#norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(resize_img)
		#return transforms.CenterCrop(size=128)(norm)
		#return transforms.ToTensor()(sample)
		#return transforms.Resize((224, 224))(tensor)


	def transform_to_ny(self, sample):
		#return transforms.Resize((256, 256))(sample)
		resize_img = transforms.Resize(size=256)(sample)
		#tensor = transforms.ToTensor()(resize_img)
		return transforms.CenterCrop(size=224)(resize_img)
		#return transforms.CenterCrop(size=224)(norm)


	# 张量数据缓存
	"""
	def cache_and_evict(self, path, target, index):

		if self.key_id_map.exists(index):
			try:
				nuy = self.key_id_map.tensorget(index)
				sample = torch.from_numpy(nuy)
			except PIL.UnidentifiedImageError:
				try:
					print("Could not open image in path from byteIO: ", path)
					sample = Image.open(path)
					sample = sample.convert('RGB')
					print("Successfully opened file from path using open.")
				except:
					print("Could not open even from path. The image file is corrupted.")
		else:
			image = Image.open(path)
			sample = image.convert('RGB')
			sample = self.transform_to_ny(sample)
			self.key_id_map.tensorset(index, sample.numpy())
		return sample
	"""

	def sort_by_imagesize(self):
		map=sorted(self.id_tensor_map.items(), key=lambda x:x[1], reverse=True)
		self.load_map = False
		self.load_max_sample(map)

	def load_max_sample(self, map):
		i=0
		for key in map:
			if i < self.cache_portion:
				path, target = self.samples[key[0]]
				self.cache_and_evict(path, target, key[0])
			i=i+1


	# 图像转为numpy缓存
	"""
	elif self.key_id_map_o.exists(index):
		try:
			byte_image = self.key_id_map_o.get(index)
			byteImgIO = io.BytesIO(byte_image)
			sample = Image.open(byteImgIO)
			sample = sample.convert('RGB')
			sample = self.transform_to_ny(sample)
			sample = transforms.ToTensor()(sample)
		except PIL.UnidentifiedImageError:
			try:
				print("Could not open image in path from byteIO: ", path)
				sample = Image.open(path)
				sample = sample.convert('RGB')
				print("Successfully opened file from path using open.")
			except:
				print("Could not open even from path. The image file is corrupted.")
	"""

	"""
	def cache_and_evict(self, path, target, index):

		if self.key_id_map.exists(index):
			try:
				nuy = self.key_id_map.tensorget(index)
				#sample = Image.fromarray(nuy)
				# for cache after resize
				#sample = torch.from_numpy(nuy)
				# for resize
				img = torch.from_numpy(nuy.transpose((2, 0, 1)))
				sample = img.float().div(255)
				#sample = Image.fromarray(nuy)
				#sample = image.convert('RGB')
			except PIL.UnidentifiedImageError:
				try:
					print("Could not open image in path from byteIO: ", path)
					sample = Image.open(path)
					sample = sample.convert('RGB')
					print("Successfully opened file from path using open.")
				except:
					print("Could not open even from path. The image file is corrupted.")
		else:
			#print("if entry here error!")
			image = Image.open(path)
			#if self.load_map == True:
			#	self.id_tensor_map[index] = imgsize[0] * imgsize[1]
				#print("================================================================================")
			#sample = image.convert('RGB')
			#byte_stream = io.BytesIO()
			#image.save(byte_stream, format=image.format)
			#byte_stream.seek(0)
			#byte_image = byte_stream.read()
			#self.key_id_map_o.set(index, byte_image)
			sample = image.convert('RGB')
			#for after decode
			sample = self.transform_to_ny(sample)
			#self.key_id_map.tensorset(index, sample.numpy())
			#if self.cur_cache_size <= self.cache_portion and self.load_map == False:
			#	#print('cur_cache_size %d, total cache size %d' % (self.cur_cache_size, self.cache_portion))
			#	self.key_id_map.tensorset(index, np.array(sample))
			#	self.cur_cache_size = self.cur_cache_size+1
			# for resize

			sample = transforms.ToTensor()(sample)
			#sample = image.convert('RGB')
		return sample
    """

	def get_map(self):
		#print(self.id_tensor_map)
		return self.id_tensor_map

	# 缓存原始数据集，使用image.save方法。
	def cache_and_evict(self, path, target, index):

		if self.key_id_map.exists(index):
			try:
				byte_image = self.key_id_map.get(index)
				byteImgIO = io.BytesIO(byte_image)
				sample = Image.open(byteImgIO)
				sample = sample.convert('RGB')
			except PIL.UnidentifiedImageError:
				try:
					print("Could not open image in path from byteIO: ", path)
					sample = Image.open(path)
					sample = sample.convert('RGB')
					print("Successfully opened file from path using open.")
				except:
					print("Could not open even from path. The image file is corrupted.")
		else:
			image = Image.open(path)
			#if self.cur_cache_size <= self.cache_portion and self.load_map == False:
			#	byte_stream = io.BytesIO()
			#	image.save(byte_stream, format=image.format)
			#	byte_stream.seek(0)
			#	byte_image = byte_stream.read()
			#	self.key_id_map.set(index, byte_image)
			#	self.cur_cache_size = self.cur_cache_size + 1
			sample = image.convert('RGB')
		return sample

	"""
	def cache_and_evict(self, path, target, index):

		if self.key_id_map.exists(index):
			try:
				nuy = self.key_id_map.tensorget(index)
				sample = torch.from_numpy(nuy)
			except PIL.UnidentifiedImageError:
				try:
					print("Could not open image in path from byteIO: ", path)
					sample = Image.open(path)
					sample = sample.convert('RGB')
					print("Successfully opened file from path using open.")
				except:
					print("Could not open even from path. The image file is corrupted.")
		else:
			image = Image.open(path)
			sample = image.convert('RGB')
			sample = self.transform_to_tensor(sample)
			self.key_id_map.tensorset(index, sample.numpy())

		return sample
	"""
	"""
	def cache_and_evict(self, path, target, index):

		if self.cache_data and self.key_id_map.exists(index):
			try:
				print('hitting %d' %(index))
				byte_image = self.key_id_map.get(index)
				byteImgIO = io.BytesIO(byte_image)
				sample = Image.open(byteImgIO)
				sample = sample.convert('RGB')
			except PIL.UnidentifiedImageError:
				try:
					print("Could not open image in path from byteIO: ", path)
					sample = Image.open(path)
					sample = sample.convert('RGB')
					print("Successfully opened file from path using open.")
				except:
					print("Could not open even from path. The image file is corrupted.")
		else:
			if index in self.ghost_cache:
				print('miss %d' %(index))
			image = Image.open(path)
			keys_cnt = self.key_counter + 50

			if(keys_cnt >= self.cache_portion):
				try:
					peek_item = self.PQ.peekitem()
					if self.ghost_cache[index] > peek_item[1]:
						evicted_item = self.PQ.popitem()
						print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))

						if self.key_id_map.exists(evicted_item[0]):
							self.key_id_map.delete(evicted_item[0])
						keys_cnt-=1
				except:
					print("Could not evict item or PQ was empty.")
					pass

			if self.cache_data and keys_cnt < self.cache_portion:
				byte_stream = io.BytesIO()
				image.save(byte_stream,format=image.format)
				byte_stream.seek(0)
				byte_image = byte_stream.read()
				self.key_id_map.set(index, byte_image)
				print("Index: ", index)
			sample = image.convert('RGB')
		return sample
	"""
	"""
	def cache_and_evict(self, path, target, index):

		if self.cache_data and self.key_id_map.exists(index):
			try:
				#print('hitting %d' %(index))
				nuy = self.key_id_map.tensorget(index)
				sample = torch.from_numpy(nuy)
				#byteImgIO = io.BytesIO(byte_image)
				#sample = torch.load(byteImgIO)
				#sample = sample.convert('RGB')
			except PIL.UnidentifiedImageError:
				try:
					print("Could not open image in path from byteIO: ", path)
					sample = Image.open(path)
					sample = sample.convert('RGB')
					sample = self.transform_to_tensor(sample)

					print("Successfully opened file from path using open.")
				except:
					print("Could not open even from path. The image file is corrupted.")
		else:
			#if index in self.ghost_cache:
				#print('miss %d' %(index))
			image = Image.open(path)
			if index not in self.id_tensor_map:
				w, h = image.size
				self.id_tensor_map[index] = w * h
			sample = image.convert('RGB')
			#print("size of the image is: %f" % sys.getsizeof(sample))
			sample = self.transform_to_tensor(sample)
			keys_cnt = self.key_counter + 50

			if(keys_cnt >= self.cache_portion):
				try:
					peek_item = self.PQ.peekitem()
					if self.ghost_cache[index] == peek_item[1]:
						evicted_item = self.PQ.popitem()
						self.evict_by_image_size(index, evicted_item[0])
					#	print(self.ghost_cache[index])
					#	print("==========================================================================")
					#	print(peek_item[1])
					#	sys.exit()
					if self.ghost_cache[index] > peek_item[1] or (self.ghost_cache[index] == peek_item[1] and self.evict_by_image_size(index, evicted_item[0])):
						#print(self.ghost_cache[index])
						#print("==========================================================================")
						#print(peek_item[1])
						evicted_item = self.PQ.popitem()
						#print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))
						#sys.exit()
						if self.key_id_map.exists(evicted_item[0]):
							self.key_id_map.delete(evicted_item[0])
						keys_cnt-=1
					else:
						print("Could not evict item or PQ was empty.")
				except Exception as ex:
					#print(peek_item)
					#print("========")
					#print(self.ghost_cache[index])
					#print("Unexpected event occurred in cache_and_evict %s."%ex)
					pass

			if self.cache_data and keys_cnt < self.cache_portion:
				#byte_stream = io.BytesIO()
				#torch.save(sample,byte_stream)
				#byte_stream.seek(0)
				#byte_tensor = byte_stream.read()
				#print("size of the decode sample is: %f" % sys.getsizeof(sample))
				#print("size of the numpy sample is: %f" % sys.getsizeof(sample.numpy()))
				#sys.exit()
				self.key_id_map.tensorset(index, sample.numpy())
				#print("Index: ", index)
		return sample

	"""

	def __getitem__(self, index: int):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (sample, target, index) where target is class_index of the target class.
		"""
		path, target = self.samples[index]
		insertion_time = datetime.now()
		insertion_time = insertion_time.strftime("%H:%M:%S")
		#print("train_search_index: %d time: %s" %(index, insertion_time))

		sample = self.cache_and_evict(path,target,index)

		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample, target, index

	def __len__(self) -> int:
		return len(self.samples)


class ShadeValDataset(Dataset):

	def __init__(self, imagefolders, transform=None, target_transform=None, cache_data = False):
		_datasets = []
		self.samples = []
		self.classes = []
		self.transform = transform
		self.target_transform = target_transform
		self.loader = None
		self.cache_data = cache_data

		for imagefolder in imagefolders:
			dataset = imagefolder
			self.loader = dataset.loader
			_datasets.append(dataset)
			self.samples.extend(dataset.samples)
			self.classes.extend(dataset.classes)
		self.classes = list(set(self.classes))

	def random_func(self):
		return 0.6858089651836363

	def __getitem__(self, index: int):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (sample, target, index) where target is class_index of the target class.
		"""
		path, target = self.samples[index]
		
		image = Image.open(path)
		sample = image.convert('RGB')

		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample, target, index

	def __len__(self) -> int:
		return len(self.samples)
