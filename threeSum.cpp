#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
using namespace std;


// 未通过
// 1. 对数组进行排序
// 2. 从数组的起始位置开始固定, 查找剩余数组中满足提交的两数
// 3. 去重，每轮查找结束，已经处理过的重复项将被跳过。
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        int len = nums.size();
        for(int i = 0; i < len-2; i++) {
            int start = i+1;
            int end = len - 1;
            if(i && nums[i] == nums[i-1]) continue;
            if (nums[i] + nums[start] + nums[end] > 0) break;
            if (nums[i] + nums[end-1] + nums[end] < 0) continue;;
            while(start < end) {
                int sum = nums[i] + nums[start] + nums[end];
                if (sum == 0) {
                    vector<int> vec = {nums[i], nums[start], nums[end]};
                    res.push_back(vec);
                    // 过滤去重
                    // for(++start; start < end; start++) {
                    //     if (nums[start] != nums[start-1]) break;;
                    // }

                    // for(--end; end > start; end--) {
                    //     if(nums[end] != nums[end+1]) break;;
                    // }
                    for (++start; start < end && nums[start] == nums[start - 1]; ++start); // 跳过重复数字
                    for (--end; end > start && nums[end] == nums[end + 1]; --end); // 跳过重复数字
                } else if (sum > 0) {
                    end--;
                } else {
                    start++;
                }
            }
            // for (++i; i < len-2; i++) {
            //     if (nums[i] != nums[i-1]) break;
            // }
        }

        return res;
        
    }
};

int main() {
    vector<int> vec = {3,0,-2,-1,1,2};
    Solution sol;

    auto res = sol.threeSum(vec);
    for(auto & vec : res) {
        for(auto &v : vec) {
            cout << v << ",";
        }

        cout << endl;
    }
}
