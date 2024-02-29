#include <memory>
#include <set>
#include <unordered_map>
#include <map>
#include <queue>
#include <iostream>
using namespace std;
// 没有完全AC
struct Node {
    Node(int k, int val, int f, int t) : key(k), value(val), frequency(f), time(t) {}
    int key;
    int value;
    int frequency;
    int time;
    // bool operator<(const Node& other) const {
    //     return ((frequency < other.frequency) || (time < other.time));
    //     // return frequency == other.frequency ? time < other.time : frequency < other.frequency;
    // }
};

struct PtrComparer {
    bool operator()(const Node* lhs, const Node* rhs) const {
        // return ((lhs->frequency < rhs->frequency) || (lhs->time < rhs->time));
        return lhs->frequency == rhs->frequency ? lhs->time < rhs->time : 
                        lhs->frequency < rhs->frequency;
    }
};

class LFUCache {
private:
    unordered_map<int, Node*> map;
    priority_queue<Node> que; // 不能用堆结构的原因是，堆查找不高效。
    set<Node*, PtrComparer> sets;
    int sequence;
    int size;

public:
    LFUCache(int capacity) : size(capacity), sequence(0) {}
    ~LFUCache() {
        for (auto it = map.begin(); it != map.end(); it++) {
            delete it->second;
        }
    }
    
    int get(int key) {
        auto iter = map.find(key);
        if (iter == map.end()) {
            return -1;
        } else {
            Node* node = map[key];
            // auto node = iter->second;
            int val = node->value;
            sets.erase(node);
            node->frequency += 1;
            node->time = ++sequence;
            sets.insert(node);
            return val;
        }
    }
    
    void put(int key, int value) {
        auto iter = map.find(key);
        if (iter != map.end()) {
            Node* node = iter->second;
            sets.erase(node);
            node->value = value;
            node->frequency += 1;
            node->time = ++sequence;
            sets.insert(node);
            cout << node->key <<","<< node->frequency << "," << node->time << "," << node->value << endl;
        } else {
            if (map.size() >= size) {
                auto it = sets.begin();
                auto key_t = (*it)->key;
                auto node = map[key_t];
                cout << "earse key is " << key_t << endl; 
                node->frequency = 1;
                node->time = ++sequence;
                node->key = key;
                node->value = value;
                map.erase(key_t);
                sets.erase(node);
                map.insert(make_pair(key, node)); 
                sets.insert(node);
            } else {
                auto *new_node = new Node(key, value, 1, sequence++);
                map.insert(make_pair(key, new_node));
                sets.insert(new_node);
            }
        }

    }
};


int main() {
    LFUCache* lfu = new LFUCache(2);
    int value;
    lfu->put(1,1);
    lfu->put(2,2);
    lfu->get(1);
    // lfu->get(1);
    // lfu->get(2);
    lfu->put(3,3); // 
    lfu->get(2); //
    lfu->get(3);
    lfu->put(4,4); 
    value=lfu->get(1);
    std::cout << value << std::endl; 

    return 0;
}
