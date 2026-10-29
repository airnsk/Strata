// Opt-in resident-state GPU correctness harness; not a throughput benchmark.
// Included only by the separately built test main, never by production main.
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace strata::ds4::test {
inline int packed_long_test(Model& m) {
    std::string err;
    auto require=[&](bool ok,const char* label){
        if(!ok) throw std::runtime_error(std::string(label)+": "+err);
    };
    struct Snapshot {std::vector<float> logits;std::vector<uint8_t> state;int draw=-1;};
    std::array<std::vector<Snapshot>,4> ref;
    for(auto& q:ref)q.resize(129);
    int comparisons=0,bad=0,top1_changes=0,draw_changes=0,state_changes=0;
    double worst_abs=0,worst_rms=0,worst_kl=0;
    const int lengths[4]={3,127,128,129};
    std::array<std::vector<int>,4> prompts;
    auto capture=[&](Snapshot& s){
        require(m.logits(s.logits),"logits");
        require(m.slot_save(s.state,err)>0,"state capture");
        for(float f:s.logits)require(std::isfinite(f),"finite logits");
        s.draw=m.sample(0.8f,0.95f,20,0.0f,123,(uint64_t)m.pos());
    };
    auto compare=[&](const Snapshot& r,int q,int stage){
        Snapshot s;capture(s);require(s.logits.size()==r.logits.size(),"logit shape");
        double mx=0,ss=0;size_t changed=0;
        for(size_t i=0;i<s.logits.size();++i){
            const double d=double(s.logits[i])-r.logits[i];mx=std::max(mx,std::abs(d));ss+=d*d;
            changed+=std::memcmp(&s.logits[i],&r.logits[i],sizeof(float))!=0;
        }
        const bool bytes=s.state==r.state;
        const double rms=std::sqrt(ss/s.logits.size());
        const auto ri=std::max_element(r.logits.begin(),r.logits.end())-r.logits.begin();
        const auto si=std::max_element(s.logits.begin(),s.logits.end())-s.logits.begin();
        double kl=0;
        if(changed) {
            double rz=0,sz=0;
            for(size_t i=0;i<s.logits.size();++i) {
                rz+=std::exp(double(r.logits[i])-r.logits[ri]);
                sz+=std::exp(double(s.logits[i])-s.logits[si]);
            }
            const double lr=double(r.logits[ri])+std::log(rz),ls=double(s.logits[si])+std::log(sz);
            for(size_t i=0;i<s.logits.size();++i)
                kl+=std::exp(double(r.logits[i])-lr)*(double(r.logits[i])-lr-double(s.logits[i])+ls);
        }
        worst_abs=std::max(worst_abs,mx);worst_rms=std::max(worst_rms,rms);worst_kl=std::max(worst_kl,kl);
        top1_changes+=ri!=si;draw_changes+=s.draw!=r.draw;state_changes+=!bytes;
        if(changed || !bytes || s.draw!=r.draw) {
            if(bad++==0) {
                if(!bytes && r.state.size()==s.state.size()) {
                    const auto u64=[&](size_t at){uint64_t v=0;if(at+8<=r.state.size())std::memcpy(&v,r.state.data()+at,8);return v;};
                    size_t at=16+(size_t)u64(8);
                    const auto n=u64(at);at+=8+(size_t)n*4;
                    for(int layer=0;layer<43 && at+8<=r.state.size();++layer) {
                        const size_t wb=(size_t)u64(at);at+=8;
                        if(at+wb>r.state.size())break;
                        if(std::memcmp(r.state.data()+at,s.state.data()+at,wb)!=0) {
                            std::fprintf(stderr,"PACKED_LONG_FIRST_STATE layer=%d field=window\n",layer);break;
                        }
                        at+=wb;if(at+8>r.state.size())break;
                        const size_t cb=(size_t)u64(at)*8;at+=8;
                        if(at+cb>r.state.size())break;
                        if(std::memcmp(r.state.data()+at,s.state.data()+at,cb)!=0) {
                            std::fprintf(stderr,"PACKED_LONG_FIRST_STATE layer=%d field=compressor\n",layer);break;
                        }
                        at+=cb;
                    }
                }
                std::fprintf(stderr,"PACKED_LONG_FIRST_DIFF request=%d stage=%d changed=%zu max=%.9g rms=%.9g KL=%.9g state_equal=%d top1_equal=%d draw_equal=%d\n",
                q,stage,changed,mx,rms,kl,int(bytes),int(ri==si),int(s.draw==r.draw));
            }
        }
        ++comparisons;
    };
    try {
        for(int q=0;q<4;++q){
            for(int i=0;i<lengths[q];++i)prompts[q].push_back(100+(i*17+q*31)%900);
            require(m.request_select(0,err),"select base");m.reset();
            require(m.feed(prompts[q].data(),lengths[q],1,err),"reference prefill");
            m.checkpoint(0);
            for(int i=0;i<128;++i){
                require(m.step(1300+q*7+i,true,err),"reference step");capture(ref[q][i]);
            }
            require(m.restore(lengths[q]),"reference restore");
            require(m.step(1700+q,true,err),"reference alternate");capture(ref[q][128]);
        }
        // Same-input B1/B1 control before attributing any difference to packing.
        require(m.request_select(0,err),"control base");m.reset();
        require(m.feed(prompts[0].data(),lengths[0],1,err),"control prefill");m.checkpoint(0);
        for(int i=0;i<128;++i){
            require(m.step(1300+i,true,err),"control step");compare(ref[0][i],0,i);
        }
        std::fprintf(stderr,"PACKED_LONG_B1_CONTROL comparisons=%d non_bit_equal=%d worst_abs=%.9g\n",comparisons,bad,worst_abs);
        require(bad==0,"B1 fixed-input control is not deterministic");
        comparisons=0;worst_abs=worst_rms=worst_kl=0;
        std::array<uint64_t,4> ids{};
        for(int q=0;q<4;++q)require(m.request_create(512,ids[q],err),"create resident");
        uint64_t excess=0;
        require(!m.request_create(512,excess,err)&&excess==0,"fifth admission rejected");err.clear();
        for(int q=0;q<4;++q){
            require(m.request_select(ids[q],err),"select resident");
            require(m.feed(prompts[q].data(),lengths[q],1,err),"resident prefill");m.checkpoint(0);
        }
        // Exercise all 24 permutations repeatedly, with mixed absolute positions.
        std::array<int,4> order={0,1,2,3};
        for(int i=0;i<128;++i){
            std::vector<uint64_t> rows;std::vector<int> tokens;
            for(int j=0;j<4;++j){int q=order[j];rows.push_back(ids[q]);tokens.push_back(1300+q*7+i);}
            require(m.decode_batch(rows,tokens,err),"packed step");
            for(int q=0;q<4;++q){
                require(m.request_select(ids[q],err),"compare select");compare(ref[q][i],q,i);
            }
            std::next_permutation(order.begin(),order.end());
        }
        for(int q=3;q>=0;--q){
            require(m.request_select(ids[q],err),"restore select");
            require(m.restore(lengths[q]),"resident restore");
            require(m.step(1700+q,true,err),"resident alternate");compare(ref[q][128],q,128);
        }
        const auto old=ids[0];require(m.request_release(old,err),"release");
        require(m.request_create(512,ids[0],err)&&ids[0]!=old,"generation reuse");
        require(!m.request_select(old,err),"stale select rejected");err.clear();
        require(m.request_select(ids[0],err)&&m.pos()==0,"fresh reused state");
        for(auto id:ids)require(m.request_release(id,err),"release all");
        require(m.request_select(0,err),"restore base binding");
        std::fprintf(stderr,"PACKED_LONG_SUMMARY comparisons=%d non_bit_equal=%d top1_changes=%d draw_changes=%d state_changes=%d worst_abs=%.9g worst_rms=%.9g worst_KL=%.9g\n",
            comparisons,bad,top1_changes,draw_changes,state_changes,worst_abs,worst_rms,worst_kl);
        if(bad){std::fputs("PACKED_LONG_NUMERICAL_GATE_FAIL\n",stderr);return 1;}
        std::fprintf(stderr,"PACKED_LONG_PASS comparisons=%d sampled_rng_equal=1\n",comparisons);return 0;
    } catch(const std::exception& x) {
        std::fprintf(stderr,"PACKED_LONG_FAIL %s\n",x.what());return 1;
    }
}
}
