#include "data0scan.h"
#include <immintrin.h>
#include <unistd.h>
#include <map>
#include <random>
#include <chrono>

#define SEL_COND(a, b) ((b >= 10 && b < 500000) && (a == 10 || a == 200 || a == 3000))
#define mm256 8

#ifdef SMID
const int con1[]  = {10, -1, 10, -1, 10, -1, 10, -1};
const int con2[]  = {200, -1, 200, -1, 200, -1, 200, -1};
const int con3[]  = {3000, -1, 3000, -1, 3000, -1, 3000, -1};
const int con4[]  = {500000, 9, 500000, 9, 500000, 9, 500000, 9};
const int con5[]  = {-1, 500000, -1, 500000, -1, 500000, -1, 500000};

__m256i condition1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(con1));
__m256i condition2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(con2));
__m256i condition3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(con3));
    
__m256i condition4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(con4));
 __m256i condition5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(con5));
 #endif

/* Default value is 2. */
static int scan_workers = 2;
/* Whether the input data in order. */
static bool row_order = false;
/* Whether to output in order according to col b.*/
static bool preserve_b_order = false;

void parallel_reader::multiway_merge_sort() {
    /* Simple hash bucket. key is a, value is an ordered interval of b. */
    map<int, node> node_list;
    for (int i = 0; i < task2_res.size(); i++) {
        node val;
        int key = task2_res[i].key;
        if (!node_list.count(key)) {
            val.push(task2_res[i]);
            node_list[key] = val;
            
        } else {
            node_list[key].push(task2_res[i]);
        }
        
    }

    for (auto iter = node_list.begin(); iter != node_list.end(); ++iter) {
        int key = iter->first;
        Temp::Arrinfo info;
        info.inx = node_list[key].top().ran.first;
        info.key = node_list[key].top().key;
        info.val_b = data->rows[info.inx].b;
        que.push(info);
    }

    while (!que.empty())
    {
        Temp::Arrinfo cur = que.top();
        que.pop();
        /* Output the current minimum element value. */
        cout << data->rows[cur.inx].a << ',' << data->rows[cur.inx].b << endl;

        Temp tmp = node_list[cur.key].top();

        if (++cur.inx <= tmp.ran.second) {
            cur.val_b = data->rows[cur.inx].b;
            que.push(cur);
        } else {
            node_list[cur.key].pop();
            /* que is not empty, skip to next interval. */
            if (!node_list[cur.key].empty()) {
                Temp::Arrinfo in = node_list[cur.key].top().ini;
                in.inx = node_list[cur.key].top().ran.first;
                in.val_b = data->rows[cur.inx].b;
                in.key = cur.key;
                que.push(in);
            }
        }
    }

    return;
}

std::pair<int, int> parallel_reader::binary_search_range_a(int target, int lf, int rt) {
    int left = lf;
    int right = rt;

    int start = -1;
    int end = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (data->rows[mid].a == target) {
            start = mid;
            right = mid - 1; 
        } else if (data->rows[mid].a < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    left = lf;
    right = rt;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (data->rows[mid].a == target) {
            end = mid;
            left = mid + 1;
        } else if (data->rows[mid].a < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return {start, end};
}

std::pair<int, int> parallel_reader::binary_search_range_b(int lf, int rt, 
                            int lowerBound, int upperBound) {
    int left = lf;
    int right = rt;

    int start = -1;
    int end = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (data->rows[mid].b >= lowerBound) {
            start = mid;
            right = mid - 1; 
        } else {
            left = mid + 1;
        }
    }

    left = lf;
    right = rt;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (data->rows[mid].b < upperBound) {
            end = mid;
            left = mid + 1; 
        } else {
            right = mid - 1;
        }
    }

    return {start, end};
}

parallel_reader::parallel_reader(const int req_n_thr, Info* in_data) : n_thread(req_n_thr), 
                        data(in_data), order(row_order), preserve_order(preserve_b_order) {
    n_thread = n_thread ? n_thread : 1;
    int range = data->nrows / n_thread;
    int start_pos = 0;
    Ranges *ran = new Ranges[n_thread]; 
    for (int cur = 0; cur < n_thread - 1; cur++) {
        ran[cur].start = start_pos;
        ran[cur].end = start_pos + range - 1;
        start_pos = start_pos + range;
    }
    ran[n_thread - 1].start = start_pos;
    ran[n_thread - 1].end = data->nrows - 1; 
    splits = ran;
}

parallel_reader::~parallel_reader() {
    delete[] splits;
    delete data;
}

void parallel_reader::process_tail(int start, int end, vector<int>* index) {
    for (int i = start; i <= end; i++) {
        int a = data->rows[i].a;
        int b = data->rows[i].b;
        if (SEL_COND(a, b)) {
            /*for tast3. */
            if (preserve_b_order) {
                index->push_back(i);
            } else {
                cout << a << ',' << b << endl;
            }
        }

    }
}

#ifdef SIMD
void parallel_reader::worker(int worker_id, bool simd) {
    cout << "SIMD" << endl;
    const int batch_size = 4;
    bool finished = false;
    vector<int>* index = new vector<int>;

    for (int pos = splits[worker_id].start; pos <= splits[worker_id].end; ) {
        if (((splits[worker_id].end - pos + 1) < batch_size)) {
            process_tail(pos, splits[worker_id].end, index);
            break;
        }
        const Row* cur = &data->rows[pos];
        simd_int_compare(cur, pos, finished, index);
        if (finished) {
            break;
        }
        pos = pos + batch_size;
    }

    /* for task3. */
    if (preserve_b_order) {
        lock();
        for(auto &inx : *index) {
            que.push(data->rows[inx]);
        }
        unlock();
    }
}
#endif

void parallel_reader::worker(int worker_id) {
    vector<int> index;
    vector<Temp> res;
    /* The row data is ordered based on (a, b). You can first do a binary search
       according to 'a' to narrow the search range, and then perform a small binary
       search according to 'b' to determine the result range.*/
    if (order) {
        vector<int> targets = {10, 200, 3000};
        Temp tmp;
        for (auto &tag: targets) {
            pair<int, int> index_a = binary_search_range_a(tag, 
                                splits[worker_id].start,  splits[worker_id].end);
            if (index_a.first != -1 && index_a.second != -1) {
                pair<int, int> index_b = binary_search_range_b(index_a.first, 
                                                        index_a.second, 10, 500000);
                if (index_b.first != -1 && index_b.second != -1) {
                    /***/
                    tmp.ran = index_b;
                    tmp.key = tag;
                    tmp.sequence_no = worker_id;
                    res.push_back(tmp);
                }
            }
        }     
    } else {
        for (int pos = splits[worker_id].start; pos <= splits[worker_id].end; pos++) {
            int a = data->rows[pos].a;
            int b = data->rows[pos].b;
            if (SEL_COND(a, b)) {
                index.push_back(pos);
            }
        }
    }

    lock();
    if (order) {
        for (auto &val : res) {
            task2_res.push_back(val);
        }
    } else {
        for(auto &inx : index) {
            vec.push_back(inx);
        }
    }
    unlock();
}

#ifdef SIMD
/* Trying to use SIMD in a worker to batch process row-based data, but the performance is not 
    as good as single data processing. */
void parallel_reader::simd_int_compare(const Row* curpos, int pos, bool &finished,
                                vector<int>* index) {
    __m256i Register = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(curpos));

    __m256i mask1 = _mm256_cmpeq_epi32(Register, condition1);
    __m256i mask2 = _mm256_cmpeq_epi32(Register, condition2);
    __m256i mask3 = _mm256_cmpeq_epi32(Register, condition3);
    __m256i lower = _mm256_cmpgt_epi32(Register, condition4);
    __m256i upper = _mm256_cmpgt_epi32(condition5, Register);

    __m256i resultMask = _mm256_or_si256(_mm256_or_si256(_mm256_or_si256(mask1, mask2), mask3), 
                    _mm256_and_si256(lower, upper));

    int resultArray[8];
    _mm256_storeu_si256((__m256i*)resultArray, resultMask);
    for (int i = 0; i < 8; i = i+2) {
            if (resultArray[i] && resultArray[i+1]) {
                cout << data->rows[pos].a <<"," << data->rows[pos].b <<endl;
            }
            pos++;
    }

}
#endif

void parallel_reader::run() {
    for (int num = 0; num < n_thread; num++) {
        // thd_list.emplace_back(&parallel_reader::worker, this, num);
        thd_list.emplace_back([this, num]() { this->worker(num); });
    }
    join();
}

void task1(const Row *rows, int nrows)
{
    Info *data = new(Info);
    data->rows = rows;
    data->nrows = nrows;
    parallel_reader *scan_reader = new parallel_reader(scan_workers, data);
    scan_reader->run();
    scan_reader->output_resut();
    delete scan_reader;

    return;
}

int main(int argc,char *argv[]) {
    int option;
    while ((option = getopt(argc, argv, "n:p:o:")) != -1) {
        switch (option) {
            case 'n':
                scan_workers = std::stoi(optarg);
                break;
            case 'p':
                preserve_b_order = std::stoi(optarg);
                break;
            case 'o':
                row_order = std::stoi(optarg);
                break;
            case '?':
                if (optopt == 'n' || optopt == 'p' || optopt == 'o') {
                    std::cerr << "Option -" << static_cast<char>(optopt) 
                              << " requires an argument." << std::endl;
                } else {
                    std::cerr << "Unknown option: -" << static_cast<char>(optopt)
                              << std::endl;
                }
                return 1; // error
            default:
                break;
        }
    }

    // std::cout << "nValue: " << scan_workers << std::endl;
    // std::cout << "pValue: " << preserve_b_order << std::endl;
    // std::cout << "oValue: " << row_order << std::endl;

    /* result:10, 100; 200, 200; 3000,10; 10,6000; 10, 499999. */
    // const vector<Row> Rows = {{10,100}, {200,200}, {1,3000}, {3000,10}, 
    // {4000,100}, {10,6000}, {10,500000}, {10,499999}}; 
    /* result : 10,31; 200,22; 200,33. */
    const vector<Row> Rows = {{ 10, 31 }, { 10, 720000000 }, { 200, 22 }, { 200, 33 }, 
                              { 1500, 12 }, { 1500, 34 }, { 3000, 5 }, };

    // const vector<Row> Rows = {{ 10, 31 }, {10, 500}, {10, 5000}, { 10, 720000000 }, 
    //                              { 200, 22 }, { 200, 33 },  { 200, 34}, {200, 499999},  
    //                              { 1500, 12 }, { 1500, 34 }, { 3000, 5 },{ 3000, 500},
    //                              { 3000, 5000}, { 3000, 5000},};


    task1(Rows.data(), Rows.size());

    return 0;
}