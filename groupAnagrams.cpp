#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>
#include <map>
#include<algorithm>
using namespace std;

// class Solution {
// public:
//     vector<vector<string>> groupAnagrams(vector<string>& strs) {
//         int size = strs.size();
//         vector<vector<string>> res;
//         for (int i = 0; i < size; i++) {
//             int cot = 0;
//             for(auto &ch : strs[i]) {
//                 cot += static_cast<int>(ch);
//             }
//             mp1.insert(make_pair(cot, i));
//         }

//         int currentkey = mp1.begin()->first;
//         for (auto iter = mp1.begin(); ; ) {
//             vector<string> vec;
//             while (iter->first == currentkey) {
//                 vec.push_back(strs[iter->second]);
//                 if (++iter == mp1.end()) {
//                     res.push_back(vec);
//                     return res;
//                 }
//             }

//             currentkey = iter->first;
//             res.push_back(vec);
//         }

//         return res;

//     }

// public:
//     multimap<int ,int> mp1;
// };

// 这道题不能简单将字符转ASCII累加转整型作为key的方式去做，因为不同字符串组合可能得到相同
// 值。
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;
        for (string& str: strs) {
            string key = str;
            sort(key.begin(), key.end());
            mp[key].emplace_back(str);
        }
        vector<vector<string>> ans;
        for (auto it = mp.begin(); it != mp.end(); ++it) {
            ans.emplace_back(it->second);
        }
        return ans;
    }
};