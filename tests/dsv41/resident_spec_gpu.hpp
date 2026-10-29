#pragma once
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
namespace strata::ds4::test {
// Compare request-owned q and segmented target verification against the same solo oracle.
inline int resident_spec_test(Model& m, bool batched=false) {
    setenv("DS4_BATCH_SPEC_TEST","1",1);
    std::string err;
    auto need=[&](bool ok,const char* s){if(!ok)throw std::runtime_error(std::string(s)+": "+err);};
    struct Round {
        std::vector<int> drafts;
        int accepted=0,next=0;
        std::vector<float> logits;
        std::vector<uint8_t> state;
    };
    std::array<std::array<Round,8>,4> refs;
    std::array<std::vector<int>,4> prompts;
    std::array<uint64_t,4> handles{},counter{};
    std::array<int,4> token{};
    const int lengths[4]={127,128,129,256};
    const float temperature=.8f,top_p=.95f,min_p=0;
    const int top_k=20;
    try {
        const char* sampled=std::getenv("DS4_DSPARK_SAMPLE");
        need(sampled && std::atoi(sampled)==1,"DS4_DSPARK_SAMPLE=1 required");
        need(m.draft_block()>0,"MTP required");
        // Initialize residency before references so both use the q ownership path.
        for(auto& h:handles)need(m.request_create(2048,h,err),"allocate");
        for(int q=0;q<4;++q) {
            for(int i=0;i<lengths[q];++i)prompts[q].push_back(100+(17*i+31*q)%900);
            need(m.request_select(0,err),"base");m.reset();
            need(m.feed(prompts[q].data(),lengths[q],1,err),"reference prefill");m.checkpoint(0);
            const uint64_t seed=1001+q;
            int y=m.sample(temperature,top_p,top_k,min_p,seed,0);uint64_t draws=1;
            for(int r=0;r<8;++r) {
                auto& ref=refs[q][r];std::vector<float> confidence;
                m.checkpoint(1);
                need(m.draft(y,ref.drafts,confidence,err,temperature,top_p,top_k,min_p,seed,draws),"reference draft");
                need(!ref.drafts.empty(),"nonempty draft required to exercise q ownership");
                need(m.verify(y,ref.drafts,temperature,top_p,top_k,min_p,seed,draws,ref.accepted,ref.next,err),"reference verify");
                need(m.logits(ref.logits),"reference logits");
                for(float x:ref.logits)need(std::isfinite(x),"finite reference");
                need(m.slot_save(ref.state,err)>0,"reference state");
                draws+=(uint64_t)ref.accepted+1;y=ref.next;
            }
        }
        for(int q=0;q<4;++q) {
            need(m.request_select(handles[q],err),"resident");
            need(m.feed(prompts[q].data(),lengths[q],1,err),"resident prefill");m.checkpoint(0);
            token[q]=m.sample(temperature,top_p,top_k,min_p,1001+q,0);counter[q]=1;
        }
        int comparisons=0;
        for(int r=0;r<8;++r) {
            std::array<std::vector<int>,4> drafts;
            // Four pending q distributions coexist before any verify consumes one.
            for(int j=0;j<4;++j) {
                int q=(j+r)%4;std::vector<float> confidence;
                need(m.request_select(handles[q],err),"draft select");m.checkpoint(1);
                need(m.draft(token[q],drafts[q],confidence,err,temperature,top_p,top_k,min_p,1001+q,counter[q]),"draft");
                need(drafts[q]==refs[q][r].drafts,"draft differs under interleave");
            }
            std::vector<SpecBatchResult> results;
            if(batched) {
                std::vector<SpecBatchRequest> requests;
                for(int j=3;j>=0;--j) {
                    const int q=(j+r)%4;
                    SpecBatchRequest x;x.request=handles[q];x.token=token[q];x.drafts=drafts[q];
                    x.temperature=temperature;x.top_p=top_p;x.top_k=top_k;x.min_p=min_p;
                    x.seed=1001+q;x.counter=counter[q];requests.push_back(std::move(x));
                }
                need(m.verify_batch(requests,results,err),"segmented verify");
                need(results.size()==4,"segmented result count");
            }
            for(int j=3;j>=0;--j) {
                int q=(j+r)%4,accepted=0,next=0;
                need(m.request_select(handles[q],err),"verify select");
                if(batched) {accepted=results[3-j].accepted;next=results[3-j].next;}
                else need(m.verify(token[q],drafts[q],temperature,top_p,top_k,min_p,1001+q,counter[q],accepted,next,err),"verify");
                std::vector<float> logits;std::vector<uint8_t> state;
                need(m.logits(logits),"logits");need(m.slot_save(state,err)>0,"state");
                const auto& ref=refs[q][r];
                const bool same=logits.size()==ref.logits.size() &&
                    !std::memcmp(logits.data(),ref.logits.data(),logits.size()*sizeof(float));
                if(!same || state!=ref.state || accepted!=ref.accepted || next!=ref.next) {
                    std::fprintf(stderr,"RESIDENT_SPEC_DIFF request=%d round=%d logits=%d state=%d accepted=%d/%d next=%d/%d\n",
                        q,r,int(same),int(state==ref.state),accepted,ref.accepted,next,ref.next);
                    throw std::runtime_error("resident speculative parity");
                }
                counter[q]+=(uint64_t)accepted+1;token[q]=next;++comparisons;
            }
        }
        for(auto h:handles)need(m.request_release(h,err),"release");
        need(m.request_select(0,err),"base restore");
        std::fprintf(stderr,"RESIDENT_SPEC_PASS comparisons=%d four_pending_q=1 serial_target_verify=%d segmented_target_verify=%d\n",comparisons,int(!batched),int(batched));
        return 0;
    } catch(const std::exception& ex) {
        std::fprintf(stderr,"RESIDENT_SPEC_FAIL %s\n",ex.what());return 1;
    }
}
}
