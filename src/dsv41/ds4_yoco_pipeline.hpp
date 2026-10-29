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
// Cone is admitted only when the entire conservative tail is in the final
// chunk. Earlier chunks retain the original source-only schedule.
inline int yoco_row_begin(int yl, int cut_k, int cut_r, bool cone, int k, int il) {
    if (yl < 0 || k != cut_k || il < yl) return 0;
    return cut_r + (cone ? (il - yl) * 127 : 0);
}
inline bool yoco_cone_last_chunk(int yl, int cut_k, int nc, int last_rows, int cut_r) {
    return yl == 20 && cut_k == nc - 1 && cut_r >= 0 && last_rows - cut_r == 2668;
}

// Dedicated gate for the measured 40-layer Flash pack, not a generic scheduler.
// Source8 crosses to layer10; every consumer chunk runs10..19 before YOCO20.
inline bool yoco_pipeline10_layout(int nl,int yl,const int* dev,const int* nm,const int* grp,
                                  const int* source,const int* ratio,const int* kv,const int* ix,const int* eg,
                                  int head,int draft,int nmtp,const int* mtpdev) {
    if(nl!=40 || yl!=20 || head!=1 || draft!=0 || nmtp!=3) return false;
    for(int i=0;i<nmtp;++i) if(mtpdev[i]!=0) return false;
    for(int il=0;il<nl;++il) {
        const int src=il<2?-1:il<8?2:il<14?8:il<20?14:20;
        const bool owner=il==2 || il==8 || il==14 || il==20;
        const bool index=owner || il==24 || il==28 || il==32 || il==36;
        const int engram=il==1?0:il==14?1:-1;
        if(nm[il]!=1 || dev[il]!=(il<10?0:1) || grp[il]!=(il<10?0:1) ||
           source[il]!=src || ratio[il]!=(il<2?0:il<20?2:1) ||
           kv[il]!=int(owner) || ix[il]!=int(index) || eg[il]!=engram) return false;
    }
    return true;
}

// The caller has already performed the dedicated split's full layout and memory gate.
inline bool yoco_cone_pipeline_admitted(int validated_source, bool pipeline15, bool pipeline10) {
    return validated_source == 20 && (pipeline15 != pipeline10);
}
}
