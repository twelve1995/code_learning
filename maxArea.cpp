#include <vector>
#include <iostream>
using namespace std;

// 简单理解先从距离最大的两根柱子开始计算面积，在读宽进行逐步缩减的过程中，踢出高度较低柱子a，
// 相比于较高 的柱子b而言，任何其他比a高的柱子与b组合，面积都大于与a组合的面积。

class Solution {
public:
    int maxArea(vector<int>& height) {
        if (height.empty() && height.size() == 1) return 0;
        int start = 0;
        int end = height.size() - 1;
        int max_area = 0;
        while (start < end)
        {
            max_area = max(max_area, (end - start) * min(height[start], height[end]));
            if (height[start] < height[end]) {
                start++;
            } else {
                end--;
            }
        }

        return max_area;
    }
};

int main() {
    vector<int> vec = {1,8,6,2,5,4,8,3,7};
    Solution sol;
    int res = sol.maxArea(vec);
    cout << res << endl;
}