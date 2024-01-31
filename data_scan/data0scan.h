#include <iostream>
#include <atomic>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
using namespace std;


/* Used to output in order according to b. */
typedef struct Temp {
    pair<int, int> ran;
    int sequence_no; /* sequence_no = worker_id, Used to ensure the local 
                        ordering of 'b'. */
    int key;
    struct Arrinfo
    {
        int key;
        int inx;    /* The current position within the interval that needs to be 
                       added to the queue.*/
        int arr_inx; /* interval index. */
        int val_b;
        bool operator>(const Arrinfo& other) const {
            return val_b > other.val_b;
        }
    } ini;

    bool operator>(const Temp& other) const {
    return sequence_no > other.sequence_no; 
    }  

} Temp;

typedef std::priority_queue<Temp, std::vector<Temp>, std::greater<Temp>>  node;

/* tuple format. */
typedef struct Row {
    int a;
    int b;

    bool operator>(const Row& other) const {
        return b > other.b;
    }
} Row;

/* The scan range of each worker. */
typedef struct Ranges {
    int start;
    int end;
} Ranges;

/* The input data information. */
typedef struct Info {
    const Row* rows;
    int nrows;
} Info;

class parallel_reader
{
private:
    int n_thread;
    bool preserve_order; /* It is expected that the matching rows 
                            printed out are sorted according to column b.*/
    bool order; /* The input parameter rows has been sorted according to (a,b) */
    Info* data;
    Ranges* splits;
    vector<std::thread> thd_list;
    /* Used to ensure that the output is ordered according to b. Protected by mutex. */
    std::priority_queue<Temp::Arrinfo, std::vector<Temp::Arrinfo>, 
                                                std::greater<Temp::Arrinfo>> que;
    /* Save result indexs, ensure correct sequence of parallel scan output results.
       Protected by mutex. */
    vector<int> vec;
    vector<Temp> task2_res;
    mutex latch;
public:
    parallel_reader() = default;
    /* constructor. */
    parallel_reader(const int req_n_thr, Info* in_data);
    /* Destructor. */
    ~parallel_reader();

    /* execute the requests. */
    void worker(int worker_id);
    void worker(int worker_id, bool simd);

    /* Start the threads to do the parallel read for the specified range. */
    void run();

    /* Use SMID to parallelize data processing and avoid multiple operator calls. 
    @param[in]   curpos         Next to read data.
    @param[in]   pos            The index of next to read data, We save the target data
                                location in the vector index.
    @param[in]   finished       Stop scanning if the currently read value of A is
                                greater than the maximum value of A (e.g. 3000).
    @param[in]   index          Save the target data location in the vector index.
    */
    void simd_int_compare(const Row* curpos, int pos, bool &finished, vector<int>* index);

    /* Process the remaining data in the splits that is not enough to be read by SMID at once. */
    void process_tail(int start, int end, vector<int>* index);

    /* Find a range that satisfies condition 'a'. */
    std::pair<int, int> binary_search_range_a(int target, int lf, int rt);
    /* Find a range that satisfies condition 'b'. */
    std::pair<int, int> binary_search_range_b(int lf, int rt, int lowerBound, int upperBound);
    /* Using hash buckets to ensure the local orderliness of b, key is a. Then use multi-way
    merge sort to ensure that the output is globally ordered according to b. The O(f(n))=
    O(n log k),  O(g(n)) = k. */
    void multiway_merge_sort();

    /* Wait for the join of threads completed by the parallel reader. */
    void join() {
        for (auto &th : thd_list) {
            th.join();
        }
    }

    void normal_print() {
        if (order) {
            return;
        }
        for (auto &inx : vec) {
            cout << data->rows[inx].a << ',' << data->rows[inx].b << endl;
        }
    }

    void task2_res_print() {
        if (!order) {
            return;
        }

        for (auto &range : task2_res) {
            for (int pos = range.ran.first; pos <= range.ran.second; pos++) {
                cout << data->rows[pos].a << ',' << data->rows[pos].b << endl;
            }
        }
    }

    void task3_res_print() {
         multiway_merge_sort();
    }

    void output_resut() {
        normal_print();
        if (preserve_order) {
            task3_res_print();
        } else {
            task2_res_print();
        }
    }

    void lock() {
        latch.lock();
    }

    void unlock() {
        latch.unlock();
    }
};