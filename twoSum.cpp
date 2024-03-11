#include <vector>
#include <unordered_map>
#include <iostream>
using namespace std;

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int size = nums.size();
        vector<int> res;
        for (int i = 0; i < size; i++) {
            mp.insert(make_pair(nums[i], i));
        }

        for (int j = 0; j < size; j++) {
            int find_key = target - nums[j];
            if (mp.count(find_key) && j != mp[find_key]) {
                res = {j, mp[target - nums[j]]};
                return res;
            }
        }

        return res;
    }

public:
    unordered_map<int, int> mp;
};

int main() {
    vector<int> vec = {2, 2, 3, 4, 5};
    Solution so;
    auto res = so.twoSum(vec, 4);
    for(auto & r : res) {
        cout << r << endl;
    }
}