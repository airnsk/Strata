#pragma once
// Experimental topology gate, not a generic multi-device YOCO scheduler.
// Every chunk still executes GPU1 layers15..19, so activation and source14
// are consumed/acknowledged even when the remainder stops at KV-only layer20.
namespace strata::ds4 {
inline bool yoco_pipeline15_layout(int nl, int yl, const int* dev, const int* nm, const int* grp) {
    if(nl!=40 || yl!=20) return false;
    for(int il=0;il<nl;++il)
        if(nm[il]!=1 || dev[il]!=(il<15?0:1) || grp[il]!=(il<15?0:1)) return false;
    return true;
}
}
