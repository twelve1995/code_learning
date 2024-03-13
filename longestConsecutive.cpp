#include <vector>
#include <set>
#include <iostream>
using namespace std;

/* 题目要求O(n)的时间复杂度，使用set遍历的时间复杂度难道不是线性的？ */
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        if (nums.empty()) {
            return 0;
        }

        for (auto &num : nums) {
            box.insert(num);
        }

        int max_seq = 0;
        int count = 1;
        auto it = box.begin();
        while (1) {
            int cur_v = (*it) + 1;
            if (++it == box.end()) {
                break;
            }
            int next_v = *(it);
            if (cur_v == next_v) {
                // cout << "result: " << (*it) << endl;
                count++;
            } else {
                if (max_seq < count) {
                    max_seq = count;
                }
                count = 1;
            }
        }

        return max_seq > count ? max_seq : count;

    }

public:
    set<int> box;
};

int main() {
    Solution sol;
    vector<int> nums = {9,1,4,7,3,-1,0,5,8,-1,6};
    int res = sol.longestConsecutive(nums);

    cout << res << endl;
}