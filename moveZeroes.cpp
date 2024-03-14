#include <vector>
#include <>
using namespace std;

// 时间复杂度接近n*n.
// class Solution {
// public:
//     void moveZeroes(vector<int>& nums) {
//         if (nums.empty()) return;
//         int size = nums.size();
//         for (int i = 0; i < size; i++) {
//             if (nums[i] != 0) {
//                 continue;
//             } else {
//                 for (int j = i; j < size; j++) {
//                     if (nums[j] != 0) {
//                         nums[i] = nums[j];
//                         nums[j] = 0;
//                         break;
//                     }
//                 }
//             }
//         }

//     }
// };

// 时间复杂度n
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        if (nums.empty()) return;
        int size = nums.size();
        int non_zero_pos = 0;
        for(int i = 0; i < size; i++) {
            if (nums[i]) {
                nums[non_zero_pos] = nums[i];
                // 发生移动需要将原始非0位置重新置为0。
                if (non_zero_pos != i) {
                    nums[i] = 0;
                }
                non_zero_pos++;
            }
        }

    }
};