#pragma once
#include <cstddef>
#include <vector>
namespace strata::ds4 {
// Preserve whole request segments and order; never split a speculative history.
inline bool spec_segment_ends(const std::vector<int>& rows,int limit,std::vector<size_t>& ends) {
    ends.clear();
    if(rows.empty() || rows.size()>4 || limit<1)return false;
    int used=0;
    for(size_t i=0;i<rows.size();++i) {
        const int n=rows[i];
        if(n<1 || n>limit) {ends.clear();return false;}
        if(n>limit-used) {ends.push_back(i);used=0;}
        used+=n;
    }
    ends.push_back(rows.size());return true;
}
}
