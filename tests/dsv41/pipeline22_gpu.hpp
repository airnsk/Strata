#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
namespace strata::ds4::test {
inline int pipeline22_benchmark(Model& m) {
    std::string err;
    auto need=[&](bool ok,const char* x){if(!ok)throw std::runtime_error(std::string(x)+": "+err);};
    std::array<uint64_t,4> ids{};
    std::array<std::vector<uint8_t>,4> initial,reference;
    std::array<std::vector<float>,4> logits_reference;
    const int lengths[4]={3,127,128,129};
    const char* old=std::getenv("DS4_BATCH_DIAG");
    const bool had_diag=old!=nullptr;const std::string diag=old?old:"";
    unsetenv("DS4_BATCH_DIAG");
    try {
        for(int q=0;q<4;++q) {
            need(m.request_create(1024,ids[q],err),"allocate");
            need(m.request_select(ids[q],err),"select");
            std::vector<int> prompt;
            for(int i=0;i<lengths[q];++i)prompt.push_back(100+(17*i+31*q)%900);
            need(m.feed(prompt.data(),(int)prompt.size(),1,err),"prefill");
            m.checkpoint(0);
            need(m.slot_save(initial[q],err)>0,"save");
        }
        const std::vector<uint64_t> rows(ids.begin(),ids.end());
        const int modes[4]={0,1,1,0};
        for(int run=0;run<4;++run) {
            setenv("DS4_BATCH_PIPELINE22",modes[run]?"1":"0",1);
            for(int q=0;q<4;++q) {
                need(m.request_select(ids[q],err),"restore select");
                need(m.slot_load(initial[q].data(),initial[q].size(),lengths[q],err),"restore");
            }
            const auto begin=std::chrono::steady_clock::now();
            for(int step=0;step<128;++step) {
                std::vector<int> tokens;
                for(int q=0;q<4;++q)tokens.push_back(1300+31*q+step);
                need(m.decode_batch(rows,tokens,err),"decode");
            }
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            for(int q=0;q<4;++q) {
                need(m.request_select(ids[q],err),"comparison select");
                std::vector<uint8_t> state;std::vector<float> logits;
                need(m.slot_save(state,err)>0,"final state");need(m.logits(logits),"final logits");
                if(run==0) {reference[q]=std::move(state);logits_reference[q]=std::move(logits);}
                else {
                    const bool same=logits.size()==logits_reference[q].size() &&
                        !std::memcmp(logits.data(),logits_reference[q].data(),logits.size()*sizeof(float));
                    if(!same || state!=reference[q]) {
                        double mx=0;
                        if(logits.size()==logits_reference[q].size())
                            for(size_t j=0;j<logits.size();++j)mx=std::max(mx,std::abs(double(logits[j])-logits_reference[q][j]));
                        std::fprintf(stderr,"PIPELINE22_BENCH_DIFF run=%d request=%d logits=%d state=%d max=%.9g\n",
                            run,q,int(same),int(state==reference[q]),mx);
                        throw std::runtime_error("pipeline final parity failed");
                    }
                }
            }
            std::fprintf(stderr,"PIPELINE22_BENCH run=%d mode=%d teacher_forced_rows=512 seconds=%.9f rows_per_s=%.9f parity=PASS\n",
                run,modes[run],seconds,512/seconds);
        }
        for(auto id:ids)need(m.request_release(id,err),"release");
        need(m.request_select(0,err),"base");
        setenv("DS4_BATCH_PIPELINE22","1",1);
        if(had_diag)setenv("DS4_BATCH_DIAG",diag.c_str(),1);
        std::fprintf(stderr,"PIPELINE22_BENCH_PASS ABBA fixed inputs; teacher-forced compute, not emitted throughput\n");
        return 0;
    } catch(const std::exception& x) {
        if(had_diag)setenv("DS4_BATCH_DIAG",diag.c_str(),1);
        std::fprintf(stderr,"PIPELINE22_BENCH_FAIL %s\n",x.what());return 1;
    }
}
}
