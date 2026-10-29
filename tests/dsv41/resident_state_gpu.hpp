// Opt-in resident-state GPU correctness harness; not a throughput benchmark.
// Included only by the separately built test main, never by production main.
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace strata::ds4::test {
inline int resident_state_test(Model& m) {
    std::string err;
    auto require=[&](bool ok,const char* label){
        if(!ok) throw std::runtime_error(std::string(label)+": "+err);
    };
    struct Snapshot {std::vector<float> logits;std::vector<uint8_t> state;};
    std::array<std::array<Snapshot,2>,4> ref;
    const int lengths[4]={3,127,128,129};
    std::array<std::vector<int>,4> prompts;
    auto capture=[&](Snapshot& s){
        require(m.logits(s.logits),"logits");
        require(m.slot_save(s.state,err)>0,"state capture");
        for(float f:s.logits)require(std::isfinite(f),"finite logits");
    };
    auto compare=[&](const Snapshot& r,int q,int stage){
        Snapshot s;capture(s);require(s.logits.size()==r.logits.size(),"logit shape");
        double mx=0,ss=0;size_t changed=0;
        for(size_t i=0;i<s.logits.size();++i){
            const double d=double(s.logits[i])-r.logits[i];mx=std::max(mx,std::abs(d));ss+=d*d;
            changed+=std::memcmp(&s.logits[i],&r.logits[i],sizeof(float))!=0;
        }
        const bool bytes=s.state==r.state;
        std::printf("RESIDENT_COMPARE request=%d stage=%d changed=%zu max=%.9g rms=%.9g state_equal=%d\n",
                    q,stage,changed,mx,std::sqrt(ss/s.logits.size()),int(bytes));std::fflush(stdout);
        require(changed==0&&bytes,"resident isolation numerical/state mismatch");
    };
    try {
        for(int q=0;q<4;++q){
            for(int i=0;i<lengths[q];++i)prompts[q].push_back(100+(i*17+q*31)%900);
            require(m.request_select(0,err),"select base");m.reset();
            require(m.feed(prompts[q].data(),lengths[q],1,err),"reference prefill");
            m.checkpoint(0);
            for(int i=0;i<3;++i)require(m.step(1300+q*7+i,true,err),"reference step");
            capture(ref[q][0]);
            require(m.restore(lengths[q]),"reference restore");
            require(m.step(1700+q,true,err),"reference alternate");capture(ref[q][1]);
        }
        std::array<uint64_t,4> ids{};
        for(int q=0;q<4;++q)require(m.request_create(256,ids[q],err),"create resident");
        uint64_t excess=0;
        require(!m.request_create(256,excess,err)&&excess==0,"fifth admission rejected");err.clear();
        for(int q=0;q<4;++q){
            require(m.request_select(ids[q],err),"select resident");
            require(m.feed(prompts[q].data(),lengths[q],1,err),"resident prefill");m.checkpoint(0);
        }
        for(int i=0;i<3;++i)for(int q=3;q>=0;--q){
            require(m.request_select(ids[q],err),"interleave select");
            require(m.step(1300+q*7+i,true,err),"interleave step");
        }
        for(int q=0;q<4;++q){require(m.request_select(ids[q],err),"compare select");compare(ref[q][0],q,0);}
        for(int q=3;q>=0;--q){
            require(m.request_select(ids[q],err),"restore select");
            require(m.restore(lengths[q]),"resident restore");
            require(m.step(1700+q,true,err),"resident alternate");compare(ref[q][1],q,1);
        }
        const auto old=ids[0];require(m.request_release(old,err),"release");
        require(m.request_create(256,ids[0],err)&&ids[0]!=old,"generation reuse");
        require(!m.request_select(old,err),"stale select rejected");err.clear();
        require(m.request_select(ids[0],err)&&m.pos()==0,"fresh reused state");
        for(auto id:ids)require(m.request_release(id,err),"release all");
        require(m.request_select(0,err),"restore base binding");
        std::puts("RESIDENT_STATE_PASS");return 0;
    } catch(const std::exception& x) {
        std::fprintf(stderr,"RESIDENT_STATE_FAIL %s\n",x.what());return 1;
    }
}
}
