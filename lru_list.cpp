#include <unordered_map>
#include <memory>
#include <iostream>

// valgrind --tool=memcheck --leak-check=full ./lru_link

struct Node
{
    Node(int k, int v) : key(k), value(v) {}
    Node() {}
    int key;
    int value;
    Node* priv;
    Node* next;
};


class LRUCache {
public:
    LRUCache(int capacity) : max_size(capacity), cur_size(0) {
        head = new Node();
        tail = new Node();
        head->next = tail;
        tail->priv = head;
    }

    ~LRUCache() {
        tail->next = nullptr;
        while (head)
        {
            auto cur = head;
            head = head->next;
            delete cur;
        }
        
    }

    void node_move(Node* node) {
        node->priv->next = node->next;
        node->next->priv = node->priv;
        add_head(node);
    }

    void add_head(Node* node) {
        node->next = head->next;
        head->next->priv = node;
        node->priv = head;
        head->next = node;
    }

    void evict_tail(Node* node) {
        node->priv->next = tail;
        tail->priv = node->priv;
    }
    
    int get(int key) {
        if (map.find(key) != map.end()) {
            auto node = map[key];
            node_move(node);
            return node->value;
        } else {
            return -1;
        }
    }
    
    void put(int key, int value) {
        if (map.find(key) != map.end()) {
            auto node = map[key];
            node->value = value;
            node_move(node);
        } else {
            if (cur_size >= max_size) {
                auto node = tail->priv;
                std::cout << "ecvit tail node key is " << node->key << std::endl;
                evict_tail(node);
                map.erase(node->key);
                node->key = key;
                node->value = value;
                // 内存复用
                node_move(node);
                map.insert({key, node});

            } else {
                // 头插即可，判断节点是否存在
                Node *node = new Node(key, value);
                add_head(node);
                map.insert({key, node});
                cur_size++;
            }
        }
    }

private:
    Node* head;
    Node* tail;
    std::unordered_map<int, Node*> map;
    int max_size;
    int cur_size;
};


int main() {
    LRUCache* lru = new LRUCache(2);
    int res = 0;
    lru->put(2,1);
    lru->put(1,1);
    res = lru->get(2);
    std::cout << res << std::endl; 
    lru->put(4,1);
    res = lru->get(1);
    std::cout << res << std::endl; 
    // lru->put(4,4);
    // res = lru->get(1);
    // std::cout << res << std::endl; 
    // res = lru->get(3);
    // std::cout << res << std::endl; 
    res = lru->get(2);
    std::cout << res << std::endl; 
    delete lru;
}