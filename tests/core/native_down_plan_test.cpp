// CPU logic test only: production inverse phases versus an independent model of
// the old per-token GPU tile. This does not establish GPU codegen or performance.
#include "strata/kernels/native_down_plan.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
namespace {
struct Metadata {
    int n = 0, ng = 0;
    std::vector<int32_t> starts, dst;
    std::vector<unsigned long long> ptr;
};
struct Image {
    std::array<int32_t,80> entry{};
    std::array<unsigned long long,80> blob{};
    std::array<uint32_t,8> bad{};
    uint32_t error = 0;
    k::NativeDownRoutePlan view() { return {entry.data(),blob.data(),bad.data()}; }
};
struct Atomics {
    template<class T> void bit_or(T* p,T value) const { *p |= value; }
    int compare_exchange(int32_t* p,int before,int after) const {
        const int old=*p; if(old==before)*p=after; return old;
    }
};
Image legacy(const Metadata& m,uint32_t prior) {
    Image out;out.error=prior;out.entry.fill(-1);
    for(int t=0;t<m.n/10;++t) {
        bool bad=m.ng<=0||m.ng>m.n;
        if(!bad)bad=m.starts[0]!=0||m.starts[m.ng]!=m.n;
        if(m.ng>0&&m.ng<=m.n)for(int g=0;g<m.ng;++g) {
            const int begin=m.starts[g],end=m.starts[g+1];
            if(begin<0||end<=begin||end>m.n||m.ptr[g]==0) {bad=true;continue;}
            for(int e=begin;e<end;++e) {
                const int d=m.dst[e];
                if(d<0||d>=m.n){bad=true;continue;}
                if(d/10==t) {
                    if(out.entry[d]!=-1)bad=true;
                    else {out.entry[d]=e;out.blob[d]=m.ptr[g];}
                }
            }
        }
        for(int d=t*10;d<(t+1)*10;++d)if(out.entry[d]<0)bad=true;
        if(bad){out.bad[t]=k::kNativeDownCombinePlanError;out.error|=k::kNativeDownCombinePlanError;}
    }
    return out;
}
void candidate(const Metadata& m,Image& out,const std::vector<int>& order) {
    const Atomics a;
    auto plan=out.view();
    for(int d=0;d<m.n;++d){plan.entry[d]=-1;plan.blob[d]=0;}
    for(int t=0;t<m.n/10;++t)plan.token_error[t]=0;
    int global_bad=m.ng<=0||m.ng>m.n;
    if(!global_bad)global_bad=m.starts[0]!=0||m.starts[m.ng]!=m.n;
    if(m.ng>0&&m.ng<=m.n)for(int g:order)if(g<m.ng)
        k::detail::native_down_plan_visit_group(g,m.ptr.data(),m.starts.data(),m.dst.data(),m.n,plan,&global_bad,a);
    for(int d=0;d<m.n;++d)k::detail::native_down_plan_check_missing(d,plan,a);
    for(int t=0;t<m.n/10;++t) {
        if(global_bad)plan.token_error[t]=k::kNativeDownCombinePlanError;
        if(plan.token_error[t])out.error|=k::kNativeDownCombinePlanError;
    }
}
Metadata valid(int tokens,std::mt19937& rng) {
    Metadata m;m.n=tokens*10;m.ng=1+int(rng()%m.n);
    m.ptr.resize(m.n);m.starts.resize(m.n+1);m.dst.resize(m.n);
    std::vector<int> cuts(m.n-1);std::iota(cuts.begin(),cuts.end(),1);
    std::shuffle(cuts.begin(),cuts.end(),rng);cuts.resize(m.ng-1);std::sort(cuts.begin(),cuts.end());
    m.starts[0]=0;std::copy(cuts.begin(),cuts.end(),m.starts.begin()+1);m.starts[m.ng]=m.n;
    for(int g=0;g<m.n;++g)m.ptr[g]=0x1000ull+static_cast<unsigned>(g)*0x200;
    std::iota(m.dst.begin(),m.dst.end(),0);std::shuffle(m.dst.begin(),m.dst.end(),rng);
    return m;
}
int cases=0;
void verify(const Metadata& m,std::mt19937& rng,Image& image) {
    const uint32_t prior=image.error;
    const auto expected=legacy(m,prior);
    const auto before=image;
    std::vector<int> order(m.n);std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
    candidate(m,image,order);
    if(image.error!=expected.error)throw std::runtime_error("aggregate OR error mismatch");
    for(int t=0;t<m.n/10;++t) {
        if(image.bad[t]!=expected.bad[t])throw std::runtime_error("token-local invalidation mismatch");
        if(!image.bad[t])for(int d=t*10;d<(t+1)*10;++d)
            if(image.entry[d]!=expected.entry[d]||image.blob[d]!=expected.blob[d])
                throw std::runtime_error("valid inverse differs from legacy tile");
    }
    for(int d=m.n;d<80;++d)if(image.entry[d]!=before.entry[d]||image.blob[d]!=before.blob[d])
        throw std::runtime_error("capacity tail overwritten");
    for(int t=m.n/10;t<8;++t)if(image.bad[t]!=before.bad[t])throw std::runtime_error("token status tail overwritten");
    ++cases;
}
}
int main() {
    try {
        std::mt19937 rng(0x90610u);
        Image image;image.entry.fill(123456);image.blob.fill(0xabcdef);image.bad.fill(0xdeadbeef);
        // Reuse a single image across changing live token counts and metadata.
        // Every malformed case is followed by valid planning, so stale errors,
        // entry indices and pointers cannot accidentally pass a single fixture.
        for(int trial=0;trial<256;++trial)for(int tokens=8;tokens>=1;--tokens) {
            const auto base=valid(tokens,rng);
            for(int fault=0;fault<14;++fault) {
                auto m=base;
                switch(fault) {
                case 0:break;
                case 1:m.ng=0;break;
                case 2:m.ng=-1;break;
                case 3:m.ng=m.n+1;break;
                case 4:m.starts[0]=1;break;
                case 5:m.starts[m.ng]=m.n-1;break;
                case 6:m.starts[rng()%m.ng]=-1;break;
                case 7:m.starts[1]=m.n+1;break;
                case 8:m.starts[1]=m.starts[0];break;
                case 9:m.ptr[rng()%m.ng]=0;break;
                case 10:m.dst[rng()%m.n]=-1;break;
                case 11:m.dst[rng()%m.n]=m.n;break;
                case 12:m.dst[0]=m.dst[1];break;
                case 13:std::fill(m.dst.begin(),m.dst.end(),0);break;
                }
                image.error=5;verify(m,rng,image);
                image.error=0;verify(base,rng,image);
            }
        }
        // A duplicate and missing slot within token 0 leave token 1 usable.
        auto local=valid(2,rng);std::iota(local.dst.begin(),local.dst.end(),0);local.dst[0]=1;
        image.error=0;verify(local,rng,image);
        if(!image.bad[0]||image.bad[1])throw std::runtime_error("local duplicate poisoned unrelated token");
        // Duplicate token 0 replacing a token 1 destination poisons both only.
        auto cross=valid(3,rng);std::iota(cross.dst.begin(),cross.dst.end(),0);cross.dst[10]=0;
        image.error=0;verify(cross,rng,image);
        if(!image.bad[0]||!image.bad[1]||image.bad[2])throw std::runtime_error("cross-token duplicate policy mismatch");
        std::printf("native_down_plan_test: %d CPU phase/oracle cases passed; GPU execution not tested\n",cases);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"native_down_plan_test: %s\n",e.what());return 1;}
}
