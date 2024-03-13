#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <iostream>
using namespace std;

// 1. 定义root节点，使用map存放字符串的起始字符地址。依次使用子节点存放字符，以及下一个关联字符
// 的子节点地址。
// 2. 搜索只要从root从上到下查找map即可；设置插入字符串的结束标识，用来判断完整字符串和前缀。

struct Node
{
    char val;
    bool falg;
    unordered_map<char, Node*> mp;
};


class Trie {
public:
    Trie() {
        root = new Node();
    }

    ~Trie() {} //释放内存，此处省略
    
    void insert(string word) {
        if (word.empty()) {
            return;
        }

        auto it = word.begin();
        /* 这里主要判断字符串或其前缀在字典中是否存在，找到断点位置进行插入即可 */
        Node* cur = root->mp[*it];
        if (!cur) {
            cur = root;
        } else {
            while(true) {
                if ((*it) == cur->val && ++it != word.end()) {
                    // 下一个字符在字典树中没有，则从断点开始插入
                    if(cur->mp.find(*it) == cur->mp.end()) {
                        break;
                    } else {
                        cur = cur->mp[*it];
                    }
                } else {
                    break;
                }
            }
        }

        while (it != word.end()) {
            Node* new_node = new Node();
            new_node->val = *it;
            cur->mp[*it] = new_node;
            cur = new_node;
            it++;
        }
        cur->falg = true;
    }
    
    bool search(string word) {
        if (word.empty()) {
            return true;
        }

        Node* cur = root;
        for (int i = 0; i < word.size(); i++) {
            if (cur->mp.find(word[i]) != cur->mp.end()) {
                cur = cur->mp[word[i]];
            }
            else {
                return false;
            }
        }

        return cur->falg ? true : false;
    }
    
    bool startsWith(string prefix) {
        if (prefix.empty()) {
            return true;
        }

        Node* cur = root;
        for (int i = 0; i < prefix.size(); i++) {
            if (cur->mp.find(prefix[i]) != cur->mp.end()) {
                cur = cur->mp[prefix[i]];
            }
            else {
                return false;
            }
        }

        return true;
    }

public: 
    Node* root;
};


int main() {
    Trie *trie = new Trie();
    trie->insert("apple");
    bool res = trie->search("app");
    res = trie->startsWith("app");
    cout << "result: " << res << endl;
}