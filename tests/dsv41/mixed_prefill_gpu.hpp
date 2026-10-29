// Test-only GPU oracle: existing requests decode while one request prefills.
// No model precision, scheduler setting, or production default is changed.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace strata::ds4::test {
inline int mixed_prefill_test(Model& m) {
    std::string err;
    auto require=[&](bool ok,const char* what){if(!ok)throw std::runtime_error(std::string(what)+": "+err);};
    struct Snapshot {std::vector<float> logits;std::vector<uint8_t> state;};
    std::array<std::vector<Snapshot>,4> ref;
    std::array<std::vector<int>,4> prompts;
    std::vector<std::vector<uint8_t>> partial;
    const int lengths[4]={3,127,129,512};
    for(auto& x:ref)x.resize(128);
    auto capture=[&](Snapshot& x){
        require(m.logits(x.logits),"logits");
        for(float f:x.logits)require(std::isfinite(f),"finite logits");
        require(m.slot_save(x.state,err)>0,"snapshot");
    };
    int comparisons=0,partial_checks=0,rows4=0;
    auto compare=[&](const Snapshot& wanted,int q,int step){
        Snapshot got;capture(got);
        const bool shape=got.logits.size()==wanted.logits.size();
        const bool same=shape && std::memcmp(got.logits.data(),wanted.logits.data(),got.logits.size()*4)==0;
        if(!same || got.state!=wanted.state) {
            double mx=0;
            if(shape)for(size_t i=0;i<got.logits.size();++i)mx=std::max(mx,std::abs(double(got.logits[i])-wanted.logits[i]));
            std::fprintf(stderr,"MIXED_PREFILL_DIFF request=%d step=%d pos=%lld logits_equal=%d state_equal=%d max=%.9g\n",
                q,step,(long long)m.pos(),int(same),int(got.state==wanted.state),mx);
            if(const char* root=std::getenv("DS4_BATCH_TEST_ARTIFACTS")) {
                const std::string prefix=std::string(root)+"/mixed-first";
                auto save=[&](const char* suffix,const void* data,size_t n){
                    std::ofstream f(prefix+suffix,std::ios::binary);f.write(static_cast<const char*>(data),n);
                    require(bool(f),"write failure artifact");
                };
                save("-reference.state",wanted.state.data(),wanted.state.size());
                save("-actual.state",got.state.data(),got.state.size());
                save("-reference.logits",wanted.logits.data(),wanted.logits.size()*4);
                save("-actual.logits",got.logits.data(),got.logits.size()*4);
            }
            throw std::runtime_error("mixed prefill state/logit mismatch");
        }
        ++comparisons;
    };
    try {
        // Same 16-token prefill partition as the native fair scheduler, on both sides.
        for(int q=0;q<4;++q) {
            for(int i=0;i<lengths[q];++i)prompts[q].push_back(100+(17*i+31*q)%900);
            require(m.request_select(0,err),"base");m.reset();
            for(int at=0;at<lengths[q];at+=16) {
                int n=std::min(16,lengths[q]-at);
                require(m.feed(prompts[q].data()+at,n,at+n==lengths[q]?1:0,err),"reference prefill quantum");
                if(q==3) {
                    partial.emplace_back();
                    require(m.slot_save(partial.back(),err)>0,"partial reference");
                }
            }
            m.checkpoint(0);
            for(int i=0;i<128;++i) {
                require(m.step(1300+31*q+i,true,err),"reference step");capture(ref[q][i]);
            }
        }
        std::array<uint64_t,4> ids{};
        for(auto& id:ids)require(m.request_create(1024,id,err),"resident allocation");
        for(int q=0;q<3;++q) {
            require(m.request_select(ids[q],err),"resident select");
            for(int at=0;at<lengths[q];at+=16) {
                int n=std::min(16,lengths[q]-at);
                require(m.feed(prompts[q].data()+at,n,at+n==lengths[q]?1:0,err),"initial resident prefill");
            }
            m.checkpoint(0);
        }
        int fed=0,steps3=0;
        for(int tick=0;tick<128;++tick) {
            std::vector<uint64_t> batch;std::vector<int> tokens,owners;
            for(int j=0;j<4;++j) {
                int q=(j+tick)%4;
                if(q==3 && fed<lengths[3])continue;
                batch.push_back(ids[q]);owners.push_back(q);
                tokens.push_back(1300+31*q+(q==3?steps3:tick));
            }
            require(m.decode_batch(batch,tokens,err),"common decode");
            rows4+=batch.size()==4;
            for(int q:owners) {
                require(m.request_select(ids[q],err),"compare request");
                compare(ref[q][q==3?steps3:tick],q,q==3?steps3:tick);
            }
            if(fed==lengths[3])++steps3;
            if(fed<lengths[3]) {
                require(m.request_select(ids[3],err),"prefill request");
                const int n=std::min(16,lengths[3]-fed);
                require(m.feed(prompts[3].data()+fed,n,fed+n==lengths[3]?1:0,err),"interleaved prefill");
                fed+=n;
                std::vector<uint8_t> state;
                require(m.slot_save(state,err)>0,"partial resident");
                require(state==partial[(size_t)fed/16-1],"interleaved partial state differs");
                ++partial_checks;
                if(fed==lengths[3])m.checkpoint(0);
            }
        }
        for(auto id:ids)require(m.request_release(id,err),"release");
        require(m.request_select(0,err),"base restore");
        require(rows4==96 && partial_checks==32,"expected mixed schedule");
        std::fprintf(stderr,"MIXED_PREFILL_PASS full_logit_state_comparisons=%d partial_states=%d common_B4_steps=%d\n",
                     comparisons,partial_checks,rows4);
        return 0;
    } catch(const std::exception& x) {
        std::fprintf(stderr,"MIXED_PREFILL_FAIL %s\n",x.what());return 1;
    }
}
}
