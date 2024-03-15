#include <string>
#include <set>
#include <iostream>
#include <map>
using namespace std;

//s = "abcabcbb"
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int max_len = 0;
        for (int i = 0; i < s.size(); i++) {
            int cur_len = 1;
            sets.clear();
            sets.insert(s[i]);
            for(int j=i+1; j < s.size(); j++) {
                if (sets.count(s[j])) {
                    break;
                } else {
                    sets.insert(s[j]);
                    cur_len++;
                }
            }

            max_len = max(max_len, cur_len);
        }

        return max_len;

    }

public:
    set<char> sets;
};

int main() {
    string s="bbbbb";
    Solution sol;
    int res = sol.lengthOfLongestSubstring(s);
    cout<< res << endl;
}