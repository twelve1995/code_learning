#include <map>
#include <unordered_map>
#include <list>
#include <memory>
#include <iostream>


class LRUCache {
public:
    LRUCache(int capacity) : max_size(capacity) {}
    
    int get(int key) {
        if (map_t.find(key) != map_t.end()) {
            auto value = *(map_t[key]);
            list_t.erase(map_t[key]);
            list_t.push_front(value);
            map_t[key] = list_t.begin();
            return value.second;
        } else {
            return -1;
        }
    }
    
    void put(int key, int value) {
        if (map_t.find(key) != map_t.end()) {
            list_t.erase(map_t[key]);
            list_t.push_front({key, value});
            map_t[key] = list_t.begin();
        } else {
            if (map_t.size() >= this->max_size) {
                auto iter = --list_t.end();
                /* list需要保存key, 用以清理map. */
                int m_key = iter->first;
                list_t.erase(iter);
                map_t.erase(m_key);
                // list_t.push_back({key, value});
                // map_t.insert({key, --list_t.end()});
                list_t.push_front({key, value});
                map_t.insert({key, list_t.begin()});
            } else {
                list_t.push_front({key, value});
                // auto addr = --list_t.end(); 
                map_t.insert({key, list_t.begin()});
            }
        }
    }

public:
    int max_size;
    std::list<std::pair<int, int>> list_t;
    std::unordered_map<int, std::list<std::pair<int, int>>::iterator> map_t;

};

int main() {
    std::shared_ptr<LRUCache> lru = std::make_shared<LRUCache>(2);
    int res = 0;
    lru->put(1,1);
    lru->put(2,2);
    res = lru->get(1);
    std::cout << res << std::endl; 
    lru->put(3,3);
    res = lru->get(2);
    std::cout << res << std::endl; 
    lru->put(4,4);
    res = lru->get(1);
    std::cout << res << std::endl; 
    res = lru->get(3);
    std::cout << res << std::endl; 
    res = lru->get(4);
    std::cout << res << std::endl;  
}