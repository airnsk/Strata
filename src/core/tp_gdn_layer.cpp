#include "strata/core/tp_gdn_layer.hpp"
#include "strata/core/tp_layer_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/tp_gdn_exchange.hpp"
#include "strata/kernels/tp_hc.hpp"
#include "strata/kernels/verify_kernels.hpp"
#if defined(STRATA_TP2_GDN_RCCL)
#include "tp_gdn_rccl.hpp"
#endif
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <new>
#include <limits>
#include <thread>
#include <chrono>
#include <cmath>
#include <atomic>
#include <cstdio>

namespace strata::core::tp2 {
namespace {
namespace k = strata::kernels;
constexpr int N=2560, F=640, HC=4, K=10, NE=512, C=10240, V=6144;
constexpr float eps=1e-6f;
void check(cudaError_t e, const char* what) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string("TP GDN ")+what+": "+cudaGetErrorString(e));
}
void select(int device) { check(cudaSetDevice(device),"select device"); }
struct CapturedNode { cudaGraph_t graph{}; cudaGraphExec_t executable{}; };
struct CapturedWindow {
    CapturedNode mixer[2], middle[4], commit[2];
    CapturedNode flat[2][2][2]; // HC split, instrumentation, physical state bank
    CapturedNode rccl[2]; // whole unprofiled hybrid proposal, physical state bank
};
struct Rank {
    TpGdnRankWeights w;
    int capacity, hk, hv, channels, value, ff, width;
    cudaStream_t compute{},copy{};
    cudaEvent_t ready{},arrival{};
    std::vector<void*> owned;
    std::vector<size_t> owned_bytes; // diagnostic snapshot extents; excludes immutable weights
    std::array<CapturedWindow,9> captured{}, captured_profile{};
    bool projection_columns() const {return tp_gdn_partition_geometry(w.partition,w.rank).projection_columns;}
    bool hybrid() const {return w.rank>=0 && w.partition==TpGdnPartition::HybridRowsColumns;}
    bool ffn_columns() const {return tp_gdn_partition_geometry(w.partition,w.rank).ffn_columns;}
    CapturedWindow& window(int t,bool profile){return profile?captured_profile[t]:captured[t];}
    int state_bank=0;
    bool resources_released=false;
    bool trace=false;
    unsigned long long* stamps{};
    float *hc_local_lo{},*hc_local_mixed{};
    std::array<uint8_t*,8> inbox{};
    k::TpGdnProtocolStamp* protocol_stamps{};
    k::TpGdnProtocolControl* control{};
    uint64_t wall_khz=0;
    float *R{},*mixed{},*attn_input{},*ffn_input{},*mixer_output{},*lo{},*rs{},*xn{},*inj[2]{},*bo{},*local_out{};
    float* peer_partial[2]{}; // distinct attention/FFN receive payloads
    float* rccl_attention_receive{}; // packed Y, then packed output; allocated only on opt-in
    float *state{},*conv{},*candidate_state{},*candidate_conv{},*qkv{},*h{},*gate{},*beta{},*z{},*y{},*full_y{};
    float *logits{},*weights{},*sg{},*su{},*scalar{},*shared{},*parts{};
    uint8_t *xq{},*shared_local_q{},*shared_full_q{},*gu_scratch{},*down_scratch{};
    int32_t *ids{},*res{},*plan{},*keep{};
    unsigned long long* offsets{};
    uint32_t* plan_error{};
    int cap_entries, ptr_offset;
    explicit Rank(const TpGdnRankWeights& weights_in,int cap):w(weights_in),capacity(cap),
        hk(w.rank<0?16:8),hv(w.rank<0?48:24),channels(w.rank<0?C:C/2),
        value(w.rank<0?V:V/2),ff(w.rank<0?F:F/2),width(tp_gdn_partition_geometry(w.partition,w.rank).ffn_output),cap_entries(cap*K) {
        select(w.device);
        try {
            check(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking),"compute stream");
            check(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking),"copy stream");
            check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming),"ready event");
            check(cudaEventCreateWithFlags(&arrival,cudaEventDisableTiming),"arrival event");
            stamps=alloc<unsigned long long>(44);
            R=alloc<float>(cap*HC*N); mixed=alloc<float>(cap*N); attn_input=alloc<float>(cap*N); ffn_input=alloc<float>(cap*N); mixer_output=alloc<float>(cap*N);
            lo=alloc<float>(cap*320);rs=alloc<float>(cap*HC);xn=alloc<float>(cap*HC*N);
            for(auto& p:inj)p=alloc<float>(cap*HC);
            bo=alloc<float>(cap*N);local_out=w.rank<0?bo:alloc<float>(cap*width);
            if(ffn_columns())for(auto& p:peer_partial)p=alloc<float>(cap*N);
            state=alloc<float>(128*hv*128);conv=alloc<float>(3*channels);
            candidate_state=alloc<float>(128*hv*128);candidate_conv=alloc<float>(3*channels);
            qkv=alloc<float>(cap*channels);h=alloc<float>(cap*channels);gate=alloc<float>(cap*hv);beta=alloc<float>(cap*hv);
            z=alloc<float>(cap*value);y=alloc<float>(cap*value);full_y=w.rank<0||projection_columns()?y:alloc<float>(cap*V);
            logits=alloc<float>(cap*NE);weights=alloc<float>(cap*K);ids=alloc<int32_t>(cap*K);
            sg=alloc<float>(cap*ff);su=alloc<float>(cap*ff);scalar=alloc<float>(cap);shared=alloc<float>(cap*width);
            parts=alloc<float>(cap*K*width);xq=alloc<uint8_t>(cap*(V/32)*36);
            shared_local_q=alloc<uint8_t>(cap*(ff/32)*36);shared_full_q=w.rank<0||ffn_columns()?shared_local_q:alloc<uint8_t>(cap*(F/32)*36);
            gu_scratch=alloc<uint8_t>(k::native_expert_scratch_bytes(cap_entries,ff));
            // Full reference and column-FFN owners consume their own GU
            // hidden in down. Only output-row TP needs a full-width gather.
            down_scratch=w.rank<0||ffn_columns()?gu_scratch:alloc<uint8_t>(k::native_expert_scratch_bytes(cap_entries,F));
            res=alloc<int32_t>(NE);offsets=alloc<unsigned long long>(NE);keep=alloc<int32_t>(1);plan_error=alloc<uint32_t>(1);
            ptr_offset=((4+(cap_entries+1)+2*cap_entries)+1)&~1;
            plan=alloc<int32_t>(ptr_offset+4*cap_entries+(cap_entries+1)+1);
            std::vector<int32_t> hr(NE);std::vector<unsigned long long> ho(NE);
            for(int i=0;i<NE;++i){hr[i]=i;ho[i]=static_cast<unsigned long long>(i)*w.expert_bytes;}
            check(cudaMemcpy(res,hr.data(),NE*sizeof(int32_t),cudaMemcpyHostToDevice),"resident table");
            check(cudaMemcpy(offsets,ho.data(),NE*sizeof(unsigned long long),cudaMemcpyHostToDevice),"expert offsets");
            // Initialization uses the default stream; nonblocking rank streams
            // must not rely on implicit legacy-default-stream dependencies.
            check(cudaDeviceSynchronize(),"initialization fence");
        } catch(...) { release();throw; }
    }
    template<class T> T* alloc(size_t n) {
        void* p=nullptr;check(cudaMalloc(&p,n*sizeof(T)),"allocation");owned.push_back(p);owned_bytes.push_back(n*sizeof(T));
        check(cudaMemset(p,0,n*sizeof(T)),"initial zero");return static_cast<T*>(p);
    }
    void destroy_graphs(bool checked = false) {
        auto destroy=[checked](CapturedNode& node){
            if(node.executable){auto e=cudaGraphExecDestroy(node.executable);if(checked)check(e,"destroy graph executable");}
            if(node.graph){auto e=cudaGraphDestroy(node.graph);if(checked)check(e,"destroy graph");}
            node={};
        };
        for(auto* windows:{&captured,&captured_profile})for(auto& window:*windows){
            for(auto& node:window.mixer)destroy(node);
            for(auto& node:window.middle)destroy(node);
            for(auto& node:window.commit)destroy(node);
            for(auto& hc:window.flat)for(auto& profile:hc)for(auto& node:profile)destroy(node);
            for(auto& node:window.rccl)destroy(node);
        }
    }
    void release(bool checked = false) {
        if(resources_released)return;
        auto status=[checked](cudaError_t e,const char* label){if(checked)check(e,label);};
        status(cudaSetDevice(w.device),"release select device");
        if(compute)status(cudaStreamSynchronize(compute),"release compute drain");
        if(copy)status(cudaStreamSynchronize(copy),"release copy drain");
        // RCCL owners destroy all graphs BEFORE their communicator is released;
        // the second call here is then a no-op. Ordinary owners retain the old order.
        destroy_graphs(checked);
        for(void* p:owned)status(cudaFree(p),"release allocation");
        owned.clear();
        if(ready)status(cudaEventDestroy(ready),"release ready event");
        if(arrival)status(cudaEventDestroy(arrival),"release arrival event");
        if(compute)status(cudaStreamDestroy(compute),"release compute stream");
        if(copy)status(cudaStreamDestroy(copy),"release copy stream");
        ready={};arrival={};compute={};copy={};
        resources_released=true;
    }
    ~Rank(){release();}
    void sync() const {select(w.device);check(cudaStreamSynchronize(compute),"rank completion");}
    void matrix(const TpNativeMatrix& m,const void* q,float* out,int t) {
        k::native_mmvq(m.type,m.data,q,out,m.input,m.output,t,compute);
    }
    void stamp(int index) {if(trace)k::gpu_stamp(stamps,index,compute);}
    void hc_args(k::FusedGrArgs* args,int half,int t,bool apply) {
        for(int j=0;j<t;++j){auto& a=args[j];a.R=R+j*HC*N;a.R_out=R+j*HC*N;a.apply=apply;
            a.bo_prev=bo+j*N;a.inj_prev=inj[0]+j*HC;a.inject_out=inj[half]+j*HC;
            a.w_norm=w.hc[half].norm;a.w_down=w.hc[half].down;a.w_up=w.hc[half].up;a.w_inject=w.hc[half].inject;
            a.lo=lo+j*320;a.rs=rs+j*HC;a.mixed=mixed+j*N;a.eps=eps;}
    }
    void hc(int half,int t,bool apply) {
        k::FusedGrArgs args[8]{};hc_args(args,half,t,apply);
        k::fused_gr_read_multi(args,t,xn,compute);
    }
    void mixer(int t,bool hc_ready=false) {
        select(w.device);if(!hc_ready)hc(0,t,false);stamp(1);
        check(cudaMemcpyAsync(attn_input,mixed,t*N*sizeof(float),cudaMemcpyDeviceToDevice,compute),"attention seam");
        k::native_quantize_q8_1(mixed,xq,N,t,compute);matrix(w.qkv,xq,qkv,t);matrix(w.z,xq,z,t);stamp(2);
        k::gdn_conv_l2_multi(conv,qkv,w.conv,h,channels,2*hk,eps,t,compute);stamp(3);
        k::gdn_ab_multi(mixed,w.alpha,w.beta,w.dt,w.a,gate,beta,N,hv,t,compute);stamp(4);
        k::gdn_step_norm_multi(state,h,channels,gate,beta,z,w.norm,eps,y,hk,hv,t,nullptr,compute);stamp(5);
    }
    void output_projection(int t) {
        // Hybrid local_out has room for T*N FFN partials, but this native
        // row projection writes packed T*(N/2), as push_output requires.
        // Never use the FFN width as the attention source token stride.
        select(w.device);k::native_quantize_q8_1(full_y,xq,w.out.input,t,compute);matrix(w.out,xq,local_out,t);stamp(9);
    }
    void ffn_gu(int t,int mode,bool hc_ready=false) {
        select(w.device);if(!hc_ready)hc(1,t,true);stamp(13);
        check(cudaMemcpyAsync(ffn_input,mixed,t*N*sizeof(float),cudaMemcpyDeviceToDevice,compute),"FFN seam");
        k::bf16_gemv_fp32_mmvf_multi(mixed,N,w.router,logits,NE,N,NE,t,compute);
        k::native_router_top10_multi(logits,ids,weights,t,compute);stamp(14);
        check(cudaMemsetAsync(plan_error,0,sizeof(uint32_t),compute),"plan error reset");
        k::resident_plan(ids,t*K,K,res,NE,w.expert_arena,offsets,static_cast<long long>(w.expert_bytes),plan,cap_entries,nullptr,0,compute,plan_error);
        k::native_quantize_q8_1(mixed,xq,N,t,compute);
        matrix(w.shared_gate,xq,sg,t);matrix(w.shared_up,xq,su,t);
        k::native_swiglu_quantize_q8_1(sg,su,shared_local_q,ff,t,compute);
        k::bf16_gemv_fp32_mmvf_multi(mixed,N,w.shared_scalar,scalar,1,N,1,t,compute);stamp(15);
        grouped(t,mode,k::NativeExpertPhase::GateUp,w.gu_layout,gu_scratch);stamp(16);
    }
    void grouped(int t,int mode,k::NativeExpertPhase phase,const k::NativeExpertLayout& layout,void* scratch) {
        const int32_t* starts=plan+4;const int32_t* dst=starts+cap_entries+1;const int32_t* tok=dst+cap_entries;
        const auto* ptr=reinterpret_cast<const unsigned long long*>(plan+ptr_offset);
        k::native_expert_grouped_explicit(layout,ptr,starts,plan,dst,tok,t*K,t*K,xq,scratch,parts,compute,{phase,mode});
    }
    uint8_t* hidden(bool full,int t) const {
        return (full?down_scratch:gu_scratch)+k::native_expert_hidden_q8_offset(t*K,full&&!ffn_columns()?F:ff);
    }
    void ffn_down(int t,int mode,float* peer_output=nullptr) {
        select(w.device);matrix(w.shared_down,shared_full_q,shared,t);stamp(20);
        if(ffn_columns()){
            const int32_t* starts=plan+4;const int32_t* dst=starts+cap_entries+1;
            const auto* ptr=reinterpret_cast<const unsigned long long*>(plan+ptr_offset);
            k::native_expert_down_combine(w.down_layout,ptr,starts,plan,dst,t*K,t*K,
                hidden(false,t),weights,shared,scalar,local_out,plan_error,t,compute,peer_output);stamp(21);stamp(22);
        }else{
            grouped(t,mode,k::NativeExpertPhase::Down,w.down_layout,down_scratch);stamp(21);
            k::native_moe_combine_multi_hits_gated(parts,weights,shared,scalar,local_out,width,K,t,compute);stamp(22);
        }
    }
    void swap_state_bank() {
        std::swap(state,candidate_state);std::swap(conv,candidate_conv);state_bank^=1;
    }
    void stage_commit(int t) {
        select(w.device);
        check(cudaMemcpyAsync(candidate_state,state,128*hv*128*sizeof(float),cudaMemcpyDeviceToDevice,compute),"stage state");
        check(cudaMemcpyAsync(candidate_conv,conv,3*channels*sizeof(float),cudaMemcpyDeviceToDevice,compute),"stage conv");
        k::gdn_conv_commit(candidate_conv,qkv,channels,keep,compute);
        k::gdn_step_norm_multi(candidate_state,h,channels,gate,beta,z,w.norm,eps,y,hk,hv,t,keep,compute,t);
    }
    template<class Launch> void capture(CapturedNode& node,Launch launch) {
        select(w.device);
        check(cudaStreamBeginCapture(compute,cudaStreamCaptureModeThreadLocal),"begin rank-local capture");
        try { launch(); }
        catch(...) {
            cudaGraph_t abandoned{};
            cudaStreamEndCapture(compute,&abandoned);
            if(abandoned)cudaGraphDestroy(abandoned);
            throw;
        }
        check(cudaStreamEndCapture(compute,&node.graph),"end rank-local capture");
        check(cudaGraphInstantiate(&node.executable,node.graph,nullptr,nullptr,0),"instantiate rank-local graph");
    }
#if defined(STRATA_TP2_GDN_RCCL)
    template<class Launch> void capture_rccl(CapturedNode& node,Launch launch) {
        // A failed RCCL capture may still own live library plans. Throw directly
        // to the paired worker boundary, which aborts/exits without freeing any
        // graph/buffer or trying to unwind a possibly blocked capture.
        select(w.device);
        check(cudaStreamBeginCapture(compute,cudaStreamCaptureModeThreadLocal),"begin RCCL capture");
        launch();
        check(cudaStreamEndCapture(compute,&node.graph),"end RCCL capture");
        check(cudaGraphInstantiate(&node.executable,node.graph,nullptr,nullptr,0),"instantiate RCCL graph");
    }
#endif
    void launch(const CapturedNode& node) {
        select(w.device);check(cudaGraphLaunch(node.executable,compute),"launch rank-local graph");
    }
    void finish(int t) {
        select(w.device);k::gr_write_multi(R,bo,inj[1],{N,HC,320},R,t,compute);stamp(26);
    }
};
}

struct TpGdnLayer::Impl {
    ModelGeometry geometry;
    std::array<std::unique_ptr<Rank>,2> r;
    int count,capacity,mode,tokens=0;
    uint64_t epoch=0;
    bool seen_epoch=false,pending=false,poisoned=false;
    TpGdnExecution execution=TpGdnExecution::RuntimeCopies;
    std::array<bool,9> prepared{},prepared_profile{};
    bool projection_columns() const {return count==2 && r[0]->projection_columns();}
    bool hybrid() const {return count==2 && r[0]->hybrid();}
    bool ffn_columns() const {return count==2 && r[0]->ffn_columns();}
    bool flat_prepared[9][2][2]{};
    uint64_t flat_timeout[9][2][2]{};
    k::TpGdnProtocolControl* host_control{};
    uint64_t protocol_epoch=0;
    bool profile_enabled=false,last_profiled=false;
    int missing_rank=-1,delayed_rank=-1;
    uint64_t delay_us=0;
    bool flat_mode() const {return execution==TpGdnExecution::FlatCaptured||execution==TpGdnExecution::FlatHcCaptured;}
    bool rccl_mode() const {return execution==TpGdnExecution::HybridRcclCaptured;}
    std::array<bool,9> rccl_prepared{};
    TpGdnRcclOptions rccl_options;
    int rccl_first=0;
#if defined(STRATA_TP2_GDN_RCCL)
    std::unique_ptr<detail::RcclSession> rccl;
#endif
    void abort_flat() noexcept {
        if(host_control)__atomic_store_n(&host_control->host_abort.value,uint64_t{1},__ATOMIC_RELEASE);
    }
    void poison_and_drain() noexcept {
        abort_flat();poisoned=true;
#if defined(STRATA_TP2_GDN_RCCL)
        if(rccl)rccl->fail("layer device failure; no output or commit is valid");
#endif
        for(auto& rank:r)if(rank){cudaSetDevice(rank->w.device);cudaStreamSynchronize(rank->compute);cudaStreamSynchronize(rank->copy);}
    }
    ~Impl(){
        abort_flat(); // release any bounded waits before owning streams drain
#if defined(STRATA_TP2_GDN_RCCL)
        if(rccl){
            rccl->shutdown([&](int rank){select(r[rank]->w.device);r[rank]->destroy_graphs(true);},
                           [&](int rank){r[rank]->release(true);});
            rccl.reset();
        }
#endif
        for(auto& rank:r)rank.reset();
#if defined(STRATA_HIP_GFX906)
        if(host_control){host_control->~TpGdnProtocolControl();hipHostFree(host_control);}
#endif
    }
    Impl(const ModelGeometry& g,const TpGdnRankWeights& a,const TpGdnRankWeights* b,int cap,int m):
        geometry(g),count(b?2:1),capacity(cap),mode(m) {
        (void)gdn_rank_layout(g,0);
        if(cap<1||cap>8)throw std::invalid_argument("TP GDN capacity must be 1..8");
        if(!k::native_expert_call_options_valid({k::NativeExpertPhase::GateUp,m}))
            throw std::invalid_argument("TP GDN invalid native expert mode");
        if((b&&(a.rank!=0||b->rank!=1||a.device==b->device||a.layer!=b->layer||a.partition!=b->partition))||(!b&&a.rank!=-1))
            throw std::invalid_argument("TP GDN needs ranks 0,1 on different devices or rank -1 reference");
        if(a.layer<0||a.layer>=g.n_layers||is_qsa_layer(g,a.layer)||a.layer==1)
            throw std::invalid_argument("TP GDN requires an ordinary non-PLE GDN layer");
        if(!k::gr_native_mmvf_enabled())throw std::invalid_argument("TP GDN requires native GR write arithmetic configured before construction");
        if(!k::shared_expert_native_bf16_enabled())throw std::invalid_argument("TP GDN requires native FP32 shared scalar-gate arithmetic");
        const char* resident_bias=std::getenv("STRATA_ROUTE_RESIDENT");
        if(resident_bias&&std::strtof(resident_bias,nullptr)!=0.0f)throw std::invalid_argument("TP GDN does not support residency-biased routing");
        auto validate=[](const TpGdnRankWeights& w) {
            const auto ownership=tp_gdn_partition_geometry(w.partition,w.rank);
            const bool full=ownership.full;
            const int c=full?C:C/2,v=full?V:V/2,f=full?F:F/2,n=ownership.ffn_output,df=ownership.ffn_input;
            auto matrix=[](const TpNativeMatrix& x,int in,int out){if(!x.data||!k::native_mmvq_supported(x.type)||x.input!=in||x.output!=out)throw std::invalid_argument("TP GDN invalid native matrix descriptor");
                (void)k::native_mmvq_weight_bytes(x.type,in,out);};
            matrix(w.qkv,N,c);matrix(w.z,N,v);matrix(w.out,ownership.projection_input,ownership.projection_output);matrix(w.shared_gate,N,f);matrix(w.shared_up,N,f);matrix(w.shared_down,df,n);
            for(const auto& h:w.hc)if(!h.norm||!h.down||!h.up||!h.inject)throw std::invalid_argument("TP GDN missing HC weights");
            if(!w.alpha||!w.beta||!w.router||!w.shared_scalar||!w.dt||!w.a||!w.conv||!w.norm||!w.expert_arena||!w.expert_bytes)
                throw std::invalid_argument("TP GDN missing layer weights");
            if(w.gu_layout.n_embd!=N||w.gu_layout.n_ff!=f||w.down_layout.n_embd!=n||w.down_layout.n_ff!=df)
                throw std::invalid_argument("TP GDN expert shape mismatch");
            const auto& gu=w.gu_layout;const auto& down=w.down_layout;
            if(!k::native_expert_supported(gu.gu_type,down.d_type,N,F)||gu.bytes!=w.expert_bytes||down.bytes!=w.expert_bytes||
               gu.gu_row!=k::iq_row_bytes(gu.gu_type,N)||down.d_row!=k::iq_row_bytes(down.d_type,df)||
               gu.up_off>gu.bytes||static_cast<size_t>(f)*gu.gu_row>gu.up_off||
               static_cast<size_t>(f)*gu.gu_row>gu.bytes-gu.up_off||
               down.down_off>down.bytes||static_cast<size_t>(n)*down.d_row>down.bytes-down.down_off)
                throw std::invalid_argument("TP GDN expert byte-span mismatch");
        };
        validate(a);if(b)validate(*b);
        if(b)for(int i=0;i<2;++i){const int src=i?b->device:a.device,dst=i?a.device:b->device;int yes=0;
            check(cudaDeviceCanAccessPeer(&yes,src,dst),"P2P query");if(!yes)throw std::runtime_error("TP GDN requires bidirectional direct P2P");
            select(src);auto e=cudaDeviceEnablePeerAccess(dst,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else check(e,"enable P2P");}
        r[0]=std::make_unique<Rank>(a,cap);if(b)r[1]=std::make_unique<Rank>(*b,cap);
        if(hybrid())execution=TpGdnExecution::HybridCaptured;
        else if(projection_columns())execution=TpGdnExecution::ColumnCaptured;
    }
    void setup_flat() {
        if(host_control)return;
#if defined(STRATA_HIP_GFX906)
        if(!k::tp_gdn_protocol_supported())throw std::runtime_error("TP GDN flat protocol primitives unavailable in this build");
        static_assert(__atomic_always_lock_free(sizeof(uint64_t),nullptr),"host signal atomics must be lock-free");
        void* memory=nullptr;
        check(hipHostMalloc(&memory,sizeof(k::TpGdnProtocolControl),hipHostMallocMapped|hipHostMallocCoherent|hipHostMallocPortable),"coherent protocol allocation");
        host_control=new(memory) k::TpGdnProtocolControl{};
        const size_t per_token[8]={160*4,1280*4,3072*4,1280*4,160*4,1280*4,11*360,1280*4};
        for(int i=0;i<count;++i){auto& a=*r[i];select(a.w.device);
            int wall=0;check(hipDeviceGetAttribute(&wall,hipDeviceAttributeWallClockRate,a.w.device),"wall clock frequency");
            // Existing gpu_stamp uses the gfx906 fixed 25 MHz wall clock.
            if(wall!=25000)throw std::runtime_error("TP GDN flat timing requires verified gfx906 25 MHz wall clock");
            a.wall_khz=static_cast<uint64_t>(wall);
            check(hipHostGetDevicePointer(reinterpret_cast<void**>(&a.control),host_control,0),"rank signal mapping");
            a.protocol_stamps=a.alloc<k::TpGdnProtocolStamp>(8);
            a.hc_local_lo=a.alloc<float>(capacity*160);a.hc_local_mixed=a.alloc<float>(capacity*1280);
            if(count==2)for(int phase=0;phase<8;++phase){void* ptr=nullptr;
                check(hipExtMallocWithFlags(&ptr,capacity*per_token[phase],hipDeviceMallocUncached),"uncached peer inbox");
                a.owned.push_back(ptr);a.owned_bytes.push_back(capacity*per_token[phase]);a.inbox[phase]=static_cast<uint8_t*>(ptr);
                check(cudaMemset(ptr,0,capacity*per_token[phase]),"initialize peer inbox");
            }
            check(cudaDeviceSynchronize(),"flat initialization fence");
        }
#else
        throw std::runtime_error("TP GDN flat protocol requires the gfx906 HIP build");
#endif
    }
    void healthy() const {if(poisoned)throw std::runtime_error("TP GDN session poisoned by prior device failure");}
    void sync() const {
#if defined(STRATA_TP2_GDN_RCCL)
        if(rccl){auto command=[&](int rank){rccl->complete(rank);};rccl->paired("layer-synchronize",std::ref(command));return;}
#endif
        for(int i=0;i<count;++i)r[i]->sync();
    }
    // Each receiving rank's copy stream waits for BOTH producers, fills disjoint
    // local slots, then releases its own consumer. No peer kernel writes/spins.
    template<class Copy> void exchange(Copy copy) {
        for(int i=0;i<count;++i){auto& a=*r[i];select(a.w.device);check(cudaEventRecord(a.ready,a.compute),"producer ready");}
        for(int dst=0;dst<count;++dst){auto& d=*r[dst];select(d.w.device);
            for(int src=0;src<count;++src)check(cudaStreamWaitEvent(d.copy,r[src]->ready,0),"wait producer");
            for(int src=0;src<count;++src)copy(d,*r[src]);
            check(cudaEventRecord(d.arrival,d.copy),"gather arrival");
            check(cudaStreamWaitEvent(d.compute,d.arrival,0),"wait gather");}
    }
    static void bytes(Rank& dst,void* to,Rank& src,const void* from,size_t n) {
        if(dst.w.device==src.w.device)check(cudaMemcpyAsync(to,from,n,cudaMemcpyDeviceToDevice,dst.copy),"local gather");
        else check(cudaMemcpyPeerAsync(to,dst.w.device,from,src.w.device,n,dst.copy),"peer gather");
    }
    void gather_y() {
        if(count==1)return; // full_y aliases y in the direct reference
        exchange([&](Rank& d,Rank& s){
            // Mapped V heads are three runs of eight 128-element heads.
            for(int t=0;t<tokens;++t)for(int b=0;b<3;++b)
                bytes(d,d.full_y+t*V+(b*16+s.w.rank*8)*128,s,s.y+t*(V/2)+b*8*128,8*128*sizeof(float));
        });
    }
    void gather_output() {
        if(count==1)return; // local_out aliases bo
        exchange([&](Rank& d,Rank& s){for(int t=0;t<tokens;++t)
            bytes(d,d.bo+t*N+(count==1?0:s.w.rank*N/2),s,s.local_out+t*s.width,s.width*sizeof(float));});
    }
    void gather_hidden() {
        if(count==1)return; // full hidden pointers alias their GU producers
        exchange([&](Rank& d,Rank& s){const size_t local=(s.ff/32)*36,full=(F/32)*36,off=count==1?0:s.w.rank*local;
            for(int e=0;e<tokens*K;++e)bytes(d,d.hidden(true,tokens)+e*full+off,s,s.hidden(false,tokens)+e*local,local);
            for(int t=0;t<tokens;++t)bytes(d,d.shared_full_q+t*full+off,s,s.shared_local_q+t*local,local);
        });
    }
    void push_y(Rank& a) {
        if(count==1)return;
        k::tp_gdn_push_y(a.y,a.full_y,r[1-a.w.rank]->full_y,tokens,a.w.rank,a.compute);
    }
    void push_output(Rank& a) {
        if(count==1)return;
        k::tp_gdn_push_output(a.local_out,a.bo,r[1-a.w.rank]->bo,tokens,a.w.rank,a.compute);
    }
    void push_hidden(Rank& a) {
        if(count==1)return;
        auto& peer=*r[1-a.w.rank];
        k::tp_gdn_push_hidden(a.hidden(false,tokens),a.hidden(true,tokens),peer.hidden(true,tokens),
            a.shared_local_q,a.shared_full_q,peer.shared_full_q,tokens,a.w.rank,a.compute);
    }
    void phase(Rank& a,int phase_index) {
        select(a.w.device);
        if(phase_index==0&&a.trace){check(cudaMemsetAsync(a.stamps,0,44*sizeof(uint64_t),a.compute),"clear captured stamps");a.stamp(0);}
        switch(phase_index){
        case 0:a.mixer(tokens);push_y(a);a.stamp(6);break;
        case 1:a.stamp(8);a.output_projection(tokens);push_output(a);a.stamp(10);break;
        case 2:
            a.stamp(12);check(cudaMemcpyAsync(a.mixer_output,a.bo,tokens*N*sizeof(float),cudaMemcpyDeviceToDevice,a.compute),"mixer seam");
            a.ffn_gu(tokens,mode);push_hidden(a);a.stamp(17);break;
        case 3:a.stamp(19);a.ffn_down(tokens,mode);push_output(a);a.stamp(23);break;
        case 4:a.stamp(25);a.finish(tokens);a.stamp(27);break;
        default:throw std::logic_error("TP GDN invalid compute phase");
        }
    }
    void phase_barrier() {
        if(count==1)return; // same-device ordered stream already establishes this
        // Record BOTH producers before submitting any consumer wait. Peer pushes
        // system-fence every writer. Both consumers wait both completed pushes,
        // including remote consumption before the next phase reuses buffers.
        for(int i=0;i<count;++i){auto& a=*r[i];select(a.w.device);check(cudaEventRecord(a.ready,a.compute),"phase producer ready");}
        for(int i=0;i<count;++i){auto& a=*r[i];select(a.w.device);
            for(int j=0;j<count;++j)check(cudaStreamWaitEvent(a.compute,r[j]->ready,0),"phase consumer wait");}
    }
    void column_reduce(Rank& a,int half) {
        if(count==1)return;
        k::tp_gdn_reduce_partials(a.w.rank==0?a.local_out:a.peer_partial[half],
            a.w.rank==0?a.peer_partial[half]:a.local_out,a.bo,tokens*N,a.compute);
    }
    void column_phase(Rank& a,int ph) {
        select(a.w.device);
        if(ph==0){
            if(a.trace){check(cudaMemsetAsync(a.stamps,0,44*sizeof(uint64_t),a.compute),"clear column stamps");a.stamp(0);}
            a.mixer(tokens);a.output_projection(tokens);
            k::tp_gdn_publish_partial(a.local_out,r[1-a.w.rank]->peer_partial[0],tokens*N,a.compute);a.stamp(10);
        }else if(ph==1){
            a.stamp(11);column_reduce(a,0);a.stamp(12);
            check(cudaMemcpyAsync(a.mixer_output,a.bo,tokens*N*sizeof(float),cudaMemcpyDeviceToDevice,a.compute),"column mixer seam");
            a.ffn_gu(tokens,mode);a.ffn_down(tokens,mode,r[1-a.w.rank]->peer_partial[1]);a.stamp(23);
        }else if(ph==2){a.stamp(24);column_reduce(a,1);a.stamp(25);a.finish(tokens);a.stamp(27);}
        else throw std::logic_error("TP GDN invalid column phase");
    }
    // Original row-owned attention association is kept through HC/router/GU.
    // Only FFN down is column-owned; no full FFN hidden buffer or gather exists.
    void hybrid_phase(Rank& a,int ph) {
        if(ph<2){phase(a,ph);return;}
        select(a.w.device);
        if(ph==2){
            a.stamp(12);
            check(cudaMemcpyAsync(a.mixer_output,a.bo,tokens*N*sizeof(float),cudaMemcpyDeviceToDevice,a.compute),"hybrid mixer seam");
            a.ffn_gu(tokens,mode);
            a.ffn_down(tokens,mode,r[1-a.w.rank]->peer_partial[1]);a.stamp(23);
        }else if(ph==3){a.stamp(24);column_reduce(a,1);a.stamp(25);a.finish(tokens);a.stamp(27);}
        else throw std::logic_error("TP GDN invalid hybrid phase");
    }
#if defined(STRATA_TP2_GDN_RCCL)
    void rccl_graph(Rank& a) {
        const int rank=a.w.rank;
        a.mixer(tokens);
        rccl->exchange(rank,a.y,a.rccl_attention_receive,size_t(tokens)*(V/2));
        k::tp_gdn_gather_y_local(a.y,a.rccl_attention_receive,a.full_y,tokens,rank,a.compute);
        a.output_projection(tokens);
        // Attention projection is packed T*1280, even though hybrid local_out
        // has capacity T*2560 for the later full-width FFN partial.
        rccl->exchange(rank,a.local_out,a.rccl_attention_receive,size_t(tokens)*(N/2));
        k::tp_gdn_gather_output_local(a.local_out,a.rccl_attention_receive,a.bo,tokens,rank,a.compute);
        check(cudaMemcpyAsync(a.mixer_output,a.bo,tokens*N*sizeof(float),cudaMemcpyDeviceToDevice,a.compute),"RCCL mixer seam");
        a.ffn_gu(tokens,mode);
        // Same fused down/combine arithmetic; transport now owns publication.
        a.ffn_down(tokens,mode,nullptr);
        rccl->exchange(rank,a.local_out,a.peer_partial[1],size_t(tokens)*N);
        column_reduce(a,1);a.finish(tokens);
    }
    template<class Input> void run_rccl(Input input) {
        if(!rccl)throw std::logic_error("TP GDN RCCL session was not prepared");
        const int omit=missing_rank,delay_rank=delayed_rank,first=rccl_first;
        const uint64_t delay=delay_us;
        missing_rank=delayed_rank=-1;delay_us=0;
        std::atomic<int> admission{0};
        const char* stage=omit>=0?"layer-missing-rank":delay_rank>=0?"layer-delayed-rank":"layer-proposal";
        auto command=[&](int rank){
            auto& a=*r[rank];select(a.w.device);
            if(rank==omit){std::printf("RCCL_LAYER_PEER_OMITTED rank=%d\n",rank);return;}
            if(rank==delay_rank&&delay)std::this_thread::sleep_for(std::chrono::microseconds(delay));
            input(a);
            // Release BEFORE entering HIP. A first launch may not return until
            // the other host thread has submitted its participating graph.
            if(omit<0){
                const int turn=rank==first?0:1;
                while(admission.load(std::memory_order_acquire)!=turn){rccl->check_peer();std::this_thread::yield();}
                admission.fetch_add(1,std::memory_order_release);
            }
            if(omit>=0||delay_rank>=0)std::printf("RCCL_LAYER_GRAPH_LAUNCH_ENTER rank=%d\n",rank);
            a.launch(a.captured[tokens].rccl[a.state_bank]);
            rccl->complete(rank);
            uint32_t error=0;
            check(cudaMemcpy(&error,a.plan_error,sizeof(error),cudaMemcpyDeviceToHost),"RCCL final plan status");
            if(error)throw std::runtime_error("TP GDN RCCL static expert plan rejected a route");
        };
        // Borrow the stack callback through the synchronous paired boundary.
        // Copying a reference_wrapper avoids heap-copying this larger closure
        // into the controller and both rank workers on every proposal.
        rccl->paired(stage,std::ref(command));
        last_profiled=false;pending=true;
    }
#endif
    void run_hybrid(bool graphs) {
        for(int ph=0;ph<4;++ph){
            for(int i=0;i<count;++i){auto& a=*r[i];
                if(graphs){auto& c=a.window(tokens,profile_enabled);a.launch(ph==0?c.mixer[a.state_bank]:c.middle[ph-1]);}
                else hybrid_phase(a,ph);
            }
            if(ph<3)phase_barrier();
        }
    }
    void run_column(bool graphs) {
        for(int ph=0;ph<3;++ph){
            for(int i=0;i<count;++i){auto& a=*r[i];
                if(graphs){auto& c=a.window(tokens,profile_enabled);a.launch(ph==0?c.mixer[a.state_bank]:c.middle[ph-1]);}
                else column_phase(a,ph);
            }
            // Each producer has already published its complete partial into
            // its peer's dedicated sublayer inbox, with a writer-system fence.
            // Both ready records precede both waits; runtime event acquisition
            // provides the same peer visibility contract as the tested pushes.
            if(ph<2)phase_barrier();
        }
    }
    void run_segmented(bool graphs) {
        if(graphs&&count==1){
            // A single GPU needs no inter-rank seams: use one complete proposal
            // graph, rather than burdening the reference with TP segmentation.
            auto& a=*r[0];a.launch(a.window(tokens,profile_enabled).mixer[a.state_bank]);return;
        }
        for(int ph=0;ph<5;++ph){
            for(int i=0;i<count;++i){auto& a=*r[i];
                if(graphs){auto& c=a.window(tokens,profile_enabled);a.launch(ph==0?c.mixer[a.state_bank]:c.middle[ph-1]);}
                else phase(a,ph);
            }
            if(ph<4)phase_barrier();
        }
    }
    void run_runtime_copies() {
        for(int i=0;i<count;++i)r[i]->mixer(tokens);
        gather_y();for(int i=0;i<count;++i)r[i]->output_projection(tokens);
        gather_output();for(int i=0;i<count;++i){auto& a=*r[i];select(a.w.device);check(cudaMemcpyAsync(a.mixer_output,a.bo,tokens*N*sizeof(float),cudaMemcpyDeviceToDevice,a.compute),"mixer seam");a.ffn_gu(tokens,mode);}
        gather_hidden();for(int i=0;i<count;++i)r[i]->ffn_down(tokens,mode);
        gather_output();for(int i=0;i<count;++i)r[i]->finish(tokens);
    }
    k::TpGdnExchangePacket packet(Rank& a,int phase) {
        k::TpGdnExchangePacket p;
        p.count=1;p.own_inbox=a.inbox[phase];p.peer_inbox=r[1-a.w.rank]->inbox[phase];
        auto& x=p.spans[0];x.rows=tokens;
        if(phase==0||phase==4){x={a.hc_local_lo,a.lo,tokens,160,160};}
        else if(phase==1||phase==5){x={a.hc_local_mixed,a.mixed,tokens,1280,1280};}
        else if(phase==2){x={a.y,a.full_y,tokens,3072,1024};}
        else if(phase==3||phase==7){x={a.local_out,a.bo,tokens,1280,1280};}
        else if(phase==6){p.count=2;x={a.hidden(false,tokens),a.hidden(true,tokens),tokens*K,90,90};
            p.spans[1]={a.shared_local_q,a.shared_full_q,tokens,90,90};}
        else throw std::logic_error("TP GDN bad protocol phase");
        for(int j=0;j<p.count;++j)p.inbox_bytes+=static_cast<size_t>(p.spans[j].rows)*p.spans[j].half_words*4;
        return p;
    }
    void flat_exchange(Rank& a,int phase,int first_stamp,uint64_t timeout_us) {
        if(count==1)return;
        const auto p=packet(a,phase);
        k::tp_gdn_packet_push(p,a.control,a.w.rank,a.compute);a.stamp(first_stamp);
        const uint64_t ticks=timeout_us*a.wall_khz/1000;
        k::tp_gdn_protocol_wait(a.control,a.protocol_stamps+phase,a.w.rank,phase,ticks,a.compute);a.stamp(first_stamp+1);
        k::tp_gdn_packet_unpack(p,a.control,a.protocol_stamps+phase,a.w.rank,phase,a.compute);a.stamp(first_stamp+2);
    }
    void flat_hc(Rank& a,int half,uint64_t timeout_us) {
        k::FusedGrArgs args[8]{};a.hc_args(args,half,tokens,half==1);
        const int phase=half==0?0:4,mark=half==0?28:36;
        k::tp_hc_down(args,tokens,a.xn,a.hc_local_lo,a.w.rank,a.compute);a.stamp(mark);
        flat_exchange(a,phase,mark+1,timeout_us);
        k::tp_hc_up(args,tokens,a.lo,a.hc_local_mixed,a.w.rank,a.compute);a.stamp(mark+4);
        flat_exchange(a,phase+1,mark+5,timeout_us);
    }
    void flat_graph(Rank& a,bool hc_split,uint64_t timeout_us) {
        select(a.w.device);
        if(a.trace)check(cudaMemsetAsync(a.stamps,0,44*sizeof(uint64_t),a.compute),"clear profile stamps");
        if(count==2)check(cudaMemsetAsync(a.protocol_stamps,0,8*sizeof(k::TpGdnProtocolStamp),a.compute),"clear protocol stamps");
        a.stamp(0);
        const bool split=hc_split&&count==2;
        if(split)flat_hc(a,0,timeout_us);
        a.mixer(tokens,split);flat_exchange(a,2,6,timeout_us);
        a.output_projection(tokens);flat_exchange(a,3,10,timeout_us);
        check(cudaMemcpyAsync(a.mixer_output,a.bo,tokens*N*sizeof(float),cudaMemcpyDeviceToDevice,a.compute),"flat mixer seam");
        if(split)flat_hc(a,1,timeout_us);
        a.ffn_gu(tokens,mode,split);flat_exchange(a,6,17,timeout_us);
        a.ffn_down(tokens,mode);flat_exchange(a,7,23,timeout_us);a.finish(tokens);a.stamp(27);
    }
    void run_flat() {
        const int hc=execution==TpGdnExecution::FlatHcCaptured;
        if(protocol_epoch==std::numeric_limits<uint64_t>::max())throw std::overflow_error("TP GDN protocol epoch exhausted");
        __atomic_store_n(&host_control->epoch.value,++protocol_epoch,__ATOMIC_RELEASE);
        const int omit=missing_rank,delay_rank=delayed_rank;const uint64_t delay=delay_us;
        missing_rank=delayed_rank=-1;delay_us=0;
        try{
            int order[2]={0,1};if(delay_rank==0)std::swap(order[0],order[1]);
            for(int j=0;j<count;++j){const int i=order[j];if(i==omit)continue;
                if(i==delay_rank&&delay)std::this_thread::sleep_for(std::chrono::microseconds(delay));
                auto& a=*r[i];a.launch(a.captured[tokens].flat[hc][profile_enabled][a.state_bank]);
            }
        }catch(...){abort_flat();throw;}
    }
    void check_flat() {
        if(__atomic_load_n(&host_control->host_abort.value,__ATOMIC_ACQUIRE)||
           __atomic_load_n(&host_control->abort[0].value,__ATOMIC_ACQUIRE)||
           __atomic_load_n(&host_control->abort[1].value,__ATOMIC_ACQUIRE))
            throw std::runtime_error("TP GDN flat protocol aborted/timed out; session poisoned, no output or commit is valid");
        if(count==1)return;
        const bool hc=execution==TpGdnExecution::FlatHcCaptured;
        // The graph has completed. Every failing wait publishes sticky abort;
        // every successful producer publishes this epoch. Read coherent host
        // words directly, avoiding additional synchronous D2H calls per window.
        for(int i=0;i<count;++i)for(int ph=0;ph<8;++ph){
            if(!hc&&(ph==0||ph==1||ph==4||ph==5))continue;
            if(__atomic_load_n(&host_control->ready[i][ph].value,__ATOMIC_ACQUIRE)!=protocol_epoch)
                throw std::runtime_error("TP GDN flat protocol incomplete epoch/phase; no commit is valid");
        }
    }
    void run() {
#if defined(STRATA_TP2_GDN_RCCL)
        if(rccl_mode()){run_rccl([](Rank&){});return;}
#endif
        if(hybrid())run_hybrid(execution==TpGdnExecution::HybridCaptured);
        else if(projection_columns())run_column(execution==TpGdnExecution::ColumnCaptured);
        else if(flat_mode())run_flat();
        else if(execution==TpGdnExecution::RuntimeCopies)run_runtime_copies();
        else run_segmented(execution==TpGdnExecution::Captured||execution==TpGdnExecution::ColumnCaptured||execution==TpGdnExecution::HybridCaptured);
        sync();
        if(flat_mode())check_flat();
        for(int i=0;i<count;++i){uint32_t error=0;select(r[i]->w.device);check(cudaMemcpy(&error,r[i]->plan_error,sizeof(error),cudaMemcpyDeviceToHost),"plan status");
            if(error)throw std::runtime_error("TP GDN static expert plan rejected a route");}
        last_profiled=profile_enabled;pending=true;
    }
    void begin(int t,uint64_t e) {
        healthy();if(pending)throw std::logic_error("TP GDN proposal requires commit(keep), including rollback with keep=0");
        if(t<1||t>capacity)throw std::invalid_argument("TP GDN token count exceeds capacity");
        if(seen_epoch&&e<=epoch)throw std::invalid_argument("TP GDN epoch must increase");
        if((execution==TpGdnExecution::Captured||execution==TpGdnExecution::ColumnCaptured||execution==TpGdnExecution::HybridCaptured)&&!(profile_enabled?prepared_profile[t]:prepared[t]))throw std::logic_error("TP GDN token count was not captured during startup");
        if(flat_mode()&&!flat_prepared[t][execution==TpGdnExecution::FlatHcCaptured][profile_enabled])
            throw std::logic_error("TP GDN flat token/profile/HC variant was not prepared during startup");
        if(rccl_mode()&&!rccl_prepared[t])throw std::logic_error("TP GDN RCCL token count was not captured during startup");
        tokens=t;epoch=e;seen_epoch=true;
    }
};

TpGdnLayer::TpGdnLayer(const ModelGeometry& g,const TpGdnRankWeights& a,const TpGdnRankWeights* b,int cap,int mode):impl_(std::make_unique<Impl>(g,a,b,cap,mode)){}
TpGdnLayer::~TpGdnLayer()=default;
void TpGdnLayer::prepare_captured(int tokens,bool profile) {
    auto& p=*impl_;p.healthy();
    if(tokens<1||tokens>p.capacity)throw std::invalid_argument("TP GDN capture token count exceeds capacity");
    if((profile?p.prepared_profile[tokens]:p.prepared[tokens]))return;
    if(p.seen_epoch||p.pending)throw std::logic_error("TP GDN capture preparation is startup-only, before the first proposal");
    if(profile)prepare_captured(tokens,false); // ordinary commit graphs are shared
    const auto old_execution=p.execution;const int old_tokens=p.tokens;const bool old_profile=p.profile_enabled;
    try {
        p.sync();p.tokens=tokens;p.execution=TpGdnExecution::Consolidated;p.profile_enabled=false;
        // Warm ordinary kernels and dispatch outside capture, without publishing
        // candidate state. Column/hybrid owners execute their own DAG here.
        p.run();p.pending=false;
        const int zero=0;
        for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);
            check(cudaMemcpyAsync(a.keep,&zero,sizeof(zero),cudaMemcpyHostToDevice,a.compute),"capture warmup keep");a.stage_commit(tokens);}
        p.sync();
        for(int i=0;i<p.count;++i){auto& a=*p.r[i];auto& window=a.window(tokens,profile);
            const int original_bank=a.state_bank;a.trace=profile;
            try{
                for(int bank=0;bank<2;++bank){
                    if(a.state_bank!=bank)a.swap_state_bank();
                    a.capture(window.mixer[bank],[&]{
                        if(p.count==1)for(int ph=0;ph<5;++ph)p.phase(a,ph);
                        else if(p.hybrid())p.hybrid_phase(a,0);
                        else if(p.projection_columns())p.column_phase(a,0);
                        else p.phase(a,0);
                    });
                    if(!profile)a.capture(window.commit[bank],[&]{a.stage_commit(tokens);});
                }
                if(a.state_bank!=original_bank)a.swap_state_bank();
                if(p.count==2)for(int ph=1;ph<(p.hybrid()?4:p.projection_columns()?3:5);++ph)
                    a.capture(window.middle[ph-1],[&]{if(p.hybrid())p.hybrid_phase(a,ph);else if(p.projection_columns())p.column_phase(a,ph);else p.phase(a,ph);});
                a.trace=false;
            }catch(...){if(a.state_bank!=original_bank)a.swap_state_bank();a.trace=false;throw;}
        }
        (profile?p.prepared_profile[tokens]:p.prepared[tokens])=true;
        p.execution=old_execution;p.tokens=old_tokens;p.profile_enabled=old_profile;
    }catch(...){p.execution=old_execution;p.tokens=old_tokens;p.profile_enabled=old_profile;p.pending=false;p.poisoned=true;throw;}
}
void TpGdnLayer::prepare_rccl(int tokens,const TpGdnRcclOptions& options) {
    auto& p=*impl_;p.healthy();
    if(!p.hybrid()||tokens<1||tokens>p.capacity||p.seen_epoch||p.pending||p.profile_enabled)
        throw std::invalid_argument("TP GDN RCCL preparation requires idle startup hybrid owners and T within capacity");
    if(options.library.empty()||options.library.front()!='/'||options.timeout_ms<100||options.timeout_ms>120000||
       options.init_timeout_ms<100||options.init_timeout_ms>120000)
        throw std::invalid_argument("TP GDN RCCL requires an explicit library and bounded 100..120000 ms timeouts");
#if defined(STRATA_TP2_GDN_RCCL)
    if(p.rccl&&(options.library!=p.rccl_options.library||options.timeout_ms!=p.rccl_options.timeout_ms||
                options.init_timeout_ms!=p.rccl_options.init_timeout_ms))
        throw std::invalid_argument("TP GDN RCCL startup options cannot change within a session");
    if(p.rccl_prepared[tokens])return;
    const int old_tokens=p.tokens;
    try{
        if(!p.rccl){
            // Warm lazy dispatch and capture ordinary commit graphs on both
            // devices before any RCCL owner thread or communicator is created.
            for(int t=1;t<=p.capacity;++t)prepare_captured(t);
            for(int rank=0;rank<2;++rank){auto& a=*p.r[rank];select(a.w.device);
                a.rccl_attention_receive=a.alloc<float>(size_t(p.capacity)*(V/2));
                check(cudaDeviceSynchronize(),"RCCL allocation initialization fence");}
            p.rccl_options=options;
            p.rccl=std::make_unique<detail::RcclSession>(
                std::array<int,2>{p.r[0]->w.device,p.r[1]->w.device},
                std::array<void*,2>{p.r[0]->compute,p.r[1]->compute},options.library,options.timeout_ms,options.init_timeout_ms);
        }
        p.tokens=tokens;
        // Actual dependent compute and all seam connections are warmed together;
        // this changes no committed state and is outside all benchmark clocks.
        p.rccl->paired("layer-warmup",[&](int rank){p.rccl_graph(*p.r[rank]);p.rccl->complete(rank);});
        p.rccl->paired("layer-capture",[&](int rank){
            auto& a=*p.r[rank];const int original=a.state_bank;
            try{
                for(int bank=0;bank<2;++bank){
                    if(a.state_bank!=bank)a.swap_state_bank();
                    a.capture_rccl(a.captured[tokens].rccl[bank],[&]{p.rccl_graph(a);});
                }
                if(a.state_bank!=original)a.swap_state_bank();
            }catch(...){if(a.state_bank!=original)a.swap_state_bank();throw;}
        },true);
        p.rccl_prepared[tokens]=true;p.tokens=old_tokens;
    }catch(...){p.tokens=old_tokens;p.pending=false;p.poison_and_drain();throw;}
#else
    (void)tokens;
    throw std::runtime_error("TP GDN RCCL was not built; configure STRATA_TP2_GDN_RCCL_BUILD=ON");
#endif
}
void TpGdnLayer::set_rccl_launch_for_test(int first,int missing,int delayed,uint64_t delay) {
    auto& p=*impl_;p.healthy();
    if(!p.rccl_mode()||p.pending||(first!=0&&first!=1)||missing < -1||missing>1||
       delayed < -1||delayed>1||delay>10000000||(missing>=0&&delayed>=0))
        throw std::invalid_argument("TP GDN RCCL launch test requires idle RCCL mode and valid rank/delay");
    p.rccl_first=first;p.missing_rank=missing;p.delayed_rank=delayed;p.delay_us=delay;
}
std::vector<TpGdnCalibrationSample> TpGdnLayer::calibrate_frozen(
        const std::vector<float>& residual,int tokens,int warmup,int trials) {
    auto& p=*impl_;p.healthy();
    if(p.pending||p.ffn_columns()||p.profile_enabled||p.flat_mode()||tokens<1||tokens>p.capacity||
       residual.size()!=size_t(tokens)*HC*N||warmup<1||trials<2)
        throw std::invalid_argument("frozen calibration requires idle unprofiled output-row/full layer and valid input");
    // Snapshot all allocations, including BOTH recurrent banks, scratch, routing
    // plan/pointers, status and injection values. No immutable weights copied.
    // Backups are host-owned: restoration traffic is deliberately outside clocks.
    struct Saved {
        Rank& a;std::vector<std::vector<uint8_t>> data;
        explicit Saved(Rank& rank):a(rank){
            select(a.w.device);a.sync();
            if(a.owned.size()!=a.owned_bytes.size())throw std::logic_error("calibration allocation manifest mismatch");
            for(size_t j=0;j<a.owned.size();++j){data.emplace_back(a.owned_bytes[j]);
                check(cudaMemcpy(data.back().data(),a.owned[j],data.back().size(),cudaMemcpyDeviceToHost),"freeze phase storage");}
        }
        void verify(const std::vector<void*>& exclude={}){select(a.w.device);
            for(size_t j=0;j<data.size();++j)if(std::find(exclude.begin(),exclude.end(),a.owned[j])==exclude.end()){
                std::vector<uint8_t> actual(data[j].size());
                check(cudaMemcpy(actual.data(),a.owned[j],actual.size(),cudaMemcpyDeviceToHost),"verify frozen allocation");
                if(actual!=data[j])throw std::runtime_error("frozen replay allocation mismatch rank="+std::to_string(a.w.rank)+" allocation="+std::to_string(j));
            }
        }
        void restore(){select(a.w.device);
            for(size_t j=0;j<data.size();++j){
                check(cudaMemcpyAsync(a.owned[j],data[j].data(),data[j].size(),cudaMemcpyHostToDevice,a.compute),"restore frozen storage");
            }
            a.sync();
        }
    };
    struct Replay {
        Rank& a;CapturedNode compute,push,empty;cudaEvent_t begin{},end{};
        explicit Replay(Rank& rank):a(rank){select(a.w.device);check(cudaEventCreate(&begin),"calibration start event");
            try{check(cudaEventCreate(&end),"calibration end event");}
            catch(...){cudaEventDestroy(begin);begin={};throw;}
        }
        ~Replay(){cudaSetDevice(a.w.device);cudaStreamSynchronize(a.compute);
            for(auto* n:{&compute,&push,&empty}){if(n->executable)cudaGraphExecDestroy(n->executable);if(n->graph)cudaGraphDestroy(n->graph);}
            if(begin)cudaEventDestroy(begin);
            if(end)cudaEventDestroy(end);}
    };
    std::array<std::unique_ptr<Saved>,2> original;
    const int old_tokens=p.tokens;const auto old_execution=p.execution;
    std::vector<TpGdnCalibrationSample> result;
    try {
        p.sync();for(int i=0;i<p.count;++i)original[i]=std::make_unique<Saved>(*p.r[i]);
        p.tokens=tokens;p.execution=TpGdnExecution::Consolidated;
        for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);
            check(cudaMemcpyAsync(a.R,residual.data(),residual.size()*sizeof(float),cudaMemcpyHostToDevice,a.compute),"calibration input");}p.sync();
        const char* names[]={"hc-attn-qkv-gdn","out-projection","hc-ffn-router-gu","down-combine","final-write"};
        for(int ph=0;ph<5;++ph){
            std::array<std::unique_ptr<Saved>,2> frozen,produced;
            std::array<std::unique_ptr<Replay>,2> replay;
            auto destinations=[&](Rank& a)->std::vector<std::pair<void*,size_t>>{
                if(p.count==1||ph==4)return {};
                if(ph==0)return {{a.full_y,size_t(tokens)*V*sizeof(float)}};
                if(ph==1||ph==3)return {{a.bo,size_t(tokens)*N*sizeof(float)}};
                return {{a.hidden(true,tokens),size_t(tokens)*K*(F/32)*36},
                        {a.shared_full_q,size_t(tokens)*(F/32)*36}};
            };
            auto excluded=[&](Rank& a)->std::vector<void*>{
                if(p.count==1||ph==4)return {};
                if(ph==0)return {a.full_y};
                if(ph==1||ph==3)return {a.bo};
                return {a.down_scratch,a.shared_full_q};
            };
            auto compute=[&](Rank& a){
                if(ph==0)a.mixer(tokens);
                else if(ph==1)a.output_projection(tokens);
                else if(ph==2){check(cudaMemcpyAsync(a.mixer_output,a.bo,tokens*N*sizeof(float),cudaMemcpyDeviceToDevice,a.compute),"calibration mixer seam");a.ffn_gu(tokens,p.mode);}
                else if(ph==3)a.ffn_down(tokens,p.mode);
                else a.finish(tokens);
            };
            auto push=[&](Rank& a){if(ph==0)p.push_y(a);else if(ph==1||ph==3)p.push_output(a);else if(ph==2)p.push_hidden(a);};
            // Materialize valid normal phase results and all gather destinations
            // before any isolated replay. The peer never runs as a prerequisite.
            for(int i=0;i<p.count;++i)frozen[i]=std::make_unique<Saved>(*p.r[i]);
            for(int i=0;i<p.count;++i)p.phase(*p.r[i],ph);
            if(ph<4)p.phase_barrier();
            p.sync();
            for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);produced[i]=std::make_unique<Saved>(a);
                replay[i]=std::make_unique<Replay>(a);a.capture(replay[i]->compute,[&]{compute(a);});
                a.capture(replay[i]->push,[&]{push(a);});a.capture(replay[i]->empty,[]{});
            }
            auto restore=[&]{for(int i=0;i<p.count;++i)frozen[i]->restore();};
            auto verify=[&](int mask,int kind){for(int i=0;i<p.count;++i)if(mask&(1<<i)){
                auto& a=*p.r[i];produced[i]->verify(kind==0?excluded(a):std::vector<void*>{});
                uint32_t error=0;select(a.w.device);check(cudaMemcpy(&error,a.plan_error,sizeof(error),cudaMemcpyDeviceToHost),"calibration plan status");
                if(error)throw std::runtime_error("calibration resident plan failed");}};
            auto measure=[&](const char* arm,int mask,int kind,int trial,int order){
                if(kind==1){
                    for(int i=0;i<p.count;++i)produced[i]->restore();
                    for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);
                        for(const auto& dst:destinations(a))check(cudaMemsetAsync(dst.first,(trial&1)?0x5a:0xa5,dst.second,a.compute),"poison gather destination");}p.sync();
                }else restore();
                // Compute end events follow their graph immediately, before another
                // rank is submitted. Joined controls end after both-rank joins.
                // Wall includes event submissions, launch(es), joins and stream
                // completion. Empty graph control exposes this measurement floor.
                const bool joined=kind!=0&&p.count==2;
                const auto start=std::chrono::steady_clock::now();
                for(int i=0;i<p.count;++i)if(mask&(1<<i)){auto& q=*replay[i];select(q.a.w.device);
                    check(cudaEventRecord(q.begin,q.a.compute),"calibration begin");
                    q.a.launch(kind==0?q.compute:kind==1?q.push:q.empty);
                    if(!joined)check(cudaEventRecord(q.end,q.a.compute),"calibration compute end");
                }
                if(joined){
                    p.phase_barrier();
                    for(int i=0;i<p.count;++i)if(mask&(1<<i)){auto& q=*replay[i];select(q.a.w.device);check(cudaEventRecord(q.end,q.a.compute),"calibration joined end");}
                }
                p.sync();const auto end=std::chrono::steady_clock::now();
                TpGdnCalibrationSample sample;sample.phase=names[ph];sample.arm=arm;sample.event_scope=joined?"graph-plus-peer-event-join":"adjacent-own-graph-events";sample.trial=trial;sample.order=order;
                sample.wall_ms=std::chrono::duration<double,std::milli>(end-start).count();
                for(int i=0;i<p.count;++i)for(size_t bytes:p.r[i]->owned_bytes)sample.restored_bytes+=bytes;
                if(kind==1){const uint64_t per_token[]={12288,5120,3960,5120,0};sample.peer_bytes_per_rank=per_token[ph]*tokens;}
                for(int i=0;i<p.count;++i)if(mask&(1<<i)){select(p.r[i]->w.device);check(cudaEventElapsedTime(&sample.device_ms[i],replay[i]->begin,replay[i]->end),"calibration event span");}
                if(kind<2)verify(mask,kind);
                if(trial>=0)result.push_back(sample);
            };
            for(int trial=-warmup;trial<trials;++trial){
                if(p.count==1){measure("full",1,0,trial,0);measure("empty-graph",1,2,trial,1);}
                else {
                    // Each crossover includes isolated and simultaneous replay
                    // in ABBA order. Alternate isolated rank order as well.
                    for(int order=0;order<4;++order){const bool alone=order==0||order==3;
                        if(alone){for(int j=0;j<2;++j){const int i=(trial&1)?1-j:j;measure(i?"half1":"half0",1<<i,0,trial,order);}}
                        else measure("concurrent-halves",3,0,trial,order);
                    }
                    if(ph<4){if(trial&1){measure("event-join-empty-graphs",3,2,trial,4);measure("push-and-event-join",3,1,trial,5);}
                        else{measure("push-and-event-join",3,1,trial,4);measure("event-join-empty-graphs",3,2,trial,5);}}
                }
            }
            // Rebuild the valid trajectory for the next phase only AFTER all
            // replays; no timed arm consumes previous replay output.
            restore();for(int i=0;i<p.count;++i)p.phase(*p.r[i],ph);
            if(ph<4)p.phase_barrier();
            p.sync();
        }
        for(int i=0;i<p.count;++i)original[i]->restore();
        for(int i=0;i<p.count;++i)original[i]->verify();
        p.tokens=old_tokens;p.execution=old_execution;
        return result;
    }catch(...){p.tokens=old_tokens;p.execution=old_execution;p.poison_and_drain();throw;}
}

std::vector<TpGdnFineCalibrationSample> TpGdnLayer::calibrate_fine_frozen(
        TpGdnLayer& full0, TpGdnLayer& full1, TpGdnLayer& hybrid,
        const std::vector<float>& residual, int tokens, int warmup, int trials) {
    auto& f0 = *full0.impl_;
    auto& f1 = *full1.impl_;
    auto& hp = *hybrid.impl_;
    const std::array<Impl*,3> owners{{&f0, &f1, &hp}};
    for (auto* owner : owners) owner->healthy();
    if (&full0 == &full1 || &full0 == &hybrid || &full1 == &hybrid ||
        f0.count != 1 || f1.count != 1 || hp.count != 2 || !hp.hybrid() ||
        f0.r[0]->w.device != hp.r[0]->w.device ||
        f1.r[0]->w.device != hp.r[1]->w.device ||
        f0.r[0]->w.layer != hp.r[0]->w.layer || f1.r[0]->w.layer != hp.r[0]->w.layer ||
        f0.mode != hp.mode || f1.mode != hp.mode ||
        (tokens != 1 && tokens != 8) || residual.size() != size_t(tokens)*HC*N ||
        warmup < 1 || warmup > 16 || trials < 2 || trials > 64)
        throw std::invalid_argument("fine calibration requires matching full0/full1/hybrid, T1/8, warmup1..16, trials2..64");
    for (auto* owner : owners) {
        if (owner->pending || owner->profile_enabled || owner->flat_mode() ||
            tokens > owner->capacity || !owner->prepared[tokens])
            throw std::invalid_argument("fine calibration requires idle unprofiled prepared layers");
    }
    std::array<Rank*,4> ranks{{f0.r[0].get(), f1.r[0].get(), hp.r[0].get(), hp.r[1].get()}};
    for (auto* rank : ranks) if (rank->trace)
        throw std::invalid_argument("fine calibration cannot use stamped ranks");
    auto sync_all = [&] { for (auto* rank : ranks) rank->sync(); };
    auto download = [](Rank& rank, const void* ptr, size_t bytes) {
        select(rank.w.device);
        std::vector<uint8_t> data(bytes);
        check(cudaMemcpy(data.data(), ptr, bytes, cudaMemcpyDeviceToHost), "fine calibration read");
        return data;
    };
    auto upload = [](Rank& rank, void* ptr, const std::vector<uint8_t>& data) {
        select(rank.w.device);
        check(cudaMemcpyAsync(ptr, data.data(), data.size(), cudaMemcpyHostToDevice, rank.compute),
              "fine calibration canonical input");
        rank.sync(); // The host buffer may be temporary; its lifetime ends here.
    };
    struct SavedFine {
        Rank& rank;
        std::vector<std::vector<uint8_t>> data;
        explicit SavedFine(Rank& r) : rank(r) {
            rank.sync();
            if (rank.owned.size() != rank.owned_bytes.size())
                throw std::logic_error("fine calibration allocation manifest mismatch");
            for (size_t j=0; j<rank.owned.size(); ++j) {
                data.emplace_back(rank.owned_bytes[j]);
                check(cudaMemcpy(data.back().data(), rank.owned[j], data.back().size(),
                                 cudaMemcpyDeviceToHost), "fine calibration freeze");
            }
        }
        void restore() {
            select(rank.w.device);
            for (size_t j=0; j<data.size(); ++j)
                check(cudaMemcpyAsync(rank.owned[j], data[j].data(), data[j].size(),
                                     cudaMemcpyHostToDevice, rank.compute), "fine calibration restore");
            rank.sync();
        }
        void verify() const {
            select(rank.w.device);
            for (size_t j=0; j<data.size(); ++j) {
                std::vector<uint8_t> actual(data[j].size());
                check(cudaMemcpy(actual.data(), rank.owned[j], actual.size(), cudaMemcpyDeviceToHost),
                      "fine calibration verify");
                if (actual != data[j])
                    throw std::runtime_error("fine calibration replay mismatch device=" +
                        std::to_string(rank.w.device) + " rank=" + std::to_string(rank.w.rank) +
                        " allocation=" + std::to_string(j));
            }
        }
    };
    struct FineGraph {
        Rank& rank;
        CapturedNode node;
        cudaEvent_t begin{}, end{};
        explicit FineGraph(Rank& r) : rank(r) {
            select(rank.w.device);
            check(cudaEventCreate(&begin), "fine calibration begin event");
            try { check(cudaEventCreate(&end), "fine calibration end event"); }
            catch (...) { cudaEventDestroy(begin); begin={}; throw; }
        }
        ~FineGraph() {
            cudaSetDevice(rank.w.device);
            cudaStreamSynchronize(rank.compute);
            if (node.executable) cudaGraphExecDestroy(node.executable);
            if (node.graph) cudaGraphDestroy(node.graph);
            if (begin) cudaEventDestroy(begin);
            if (end) cudaEventDestroy(end);
        }
    };
    std::array<std::unique_ptr<SavedFine>,4> original;
    const std::array<int,3> old_tokens{{f0.tokens, f1.tokens, hp.tokens}};
    std::vector<TpGdnFineCalibrationSample> result;
    uint64_t restored_bytes = 0;
    for (auto* rank : ranks) for (size_t bytes : rank->owned_bytes) restored_bytes += bytes;
    int groups = 0;
    std::vector<int> group_entries;
    // The checksum identifies canonical input bytes within this fixture, not
    // model provenance or owner-specific resident pointers.
    uint64_t input_hash = 14695981039346656037ull;
    auto hash_bytes = [&](const std::vector<uint8_t>& data) {
        for (uint8_t value : data) { input_hash ^= value; input_hash *= 1099511628211ull; }
    };
    auto canonical = [&](auto pointer, size_t bytes) {
        auto data = download(*ranks[0], pointer(*ranks[0]), bytes);
        hash_bytes(data);
        for (int i=1; i<4; ++i) upload(*ranks[i], pointer(*ranks[i]), data);
        return data;
    };
    std::vector<uint8_t> canonical_routed_hidden, canonical_shared_hidden;
    auto sliced_hidden = [&](bool routed) {
        const size_t rows = size_t(tokens)*(routed ? K : 1);
        const size_t full_row = (F/32)*36, half_row = full_row/2;
        auto pointer = [&](Rank& rank) -> uint8_t* {
            return routed ? rank.hidden(false,tokens) : rank.shared_local_q;
        };
        const auto& data = routed ? canonical_routed_hidden : canonical_shared_hidden;
        if (data.size()!=rows*full_row) throw std::logic_error("fine calibration canonical hidden shape");
        hash_bytes(data);
        upload(*ranks[0], pointer(*ranks[0]), data);
        upload(*ranks[1], pointer(*ranks[1]), data);
        for (int i=2; i<4; ++i) {
            std::vector<uint8_t> half(rows*half_row);
            for (size_t row=0; row<rows; ++row)
                std::copy_n(data.data()+row*full_row+(i-2)*half_row, half_row, half.data()+row*half_row);
            upload(*ranks[i], pointer(*ranks[i]), half);
        }
    };
    auto exact_canonical = [&](auto pointer, size_t bytes, const char* label) {
        const auto expected = download(*ranks[0], pointer(*ranks[0]), bytes);
        for (int i=1; i<4; ++i) if (download(*ranks[i], pointer(*ranks[i]), bytes) != expected)
            throw std::runtime_error(std::string("fine calibration canonical mismatch: ") + label);
    };
    auto exact_hidden = [&](bool routed) {
        const size_t rows = size_t(tokens)*(routed ? K : 1);
        const size_t full_row = (F/32)*36, half_row = full_row/2;
        auto pointer = [&](Rank& rank) { return routed ? rank.hidden(false,tokens) : rank.shared_local_q; };
        const auto expected = download(*ranks[0], pointer(*ranks[0]), rows*full_row);
        if (download(*ranks[1], pointer(*ranks[1]), rows*full_row) != expected)
            throw std::runtime_error("fine calibration full-device hidden mismatch");
        for (int i=2; i<4; ++i) {
            const auto half = download(*ranks[i], pointer(*ranks[i]), rows*half_row);
            for (size_t row=0; row<rows; ++row)
                if (std::memcmp(half.data()+row*half_row,
                                expected.data()+row*full_row+(i-2)*half_row, half_row))
                    throw std::runtime_error("fine calibration canonical hidden ownership mismatch");
        }
    };
    auto numeric_gate = [](const std::vector<uint8_t>& reference,
                           const std::vector<uint8_t>& actual, const char* label) {
        if (reference.empty() || reference.size()!=actual.size() || reference.size()%sizeof(float))
            throw std::logic_error("fine calibration numerical gate shape");
        double square=0, error=0, maximum=0;
        for (size_t j=0; j<reference.size(); j+=sizeof(float)) {
            float expected_value, actual_value;
            std::memcpy(&expected_value,reference.data()+j,sizeof(float));
            std::memcpy(&actual_value,actual.data()+j,sizeof(float));
            if (!std::isfinite(expected_value) || !std::isfinite(actual_value))
                throw std::runtime_error(std::string("fine calibration nonfinite ")+label);
            const double delta=double(expected_value)-actual_value;
            square+=double(expected_value)*expected_value; error+=delta*delta;
            maximum=std::max(maximum,std::abs(delta));
        }
        const double count=double(reference.size()/sizeof(float)), scale=std::sqrt(square/count);
        // Unchanged ffn_gate from the targeted full-layer fixture.
        if (maximum>1e-5+3e-4*scale || std::sqrt(error/count)>1e-5+5e-5*scale)
            throw std::runtime_error(std::string("fine calibration numerical gate: ")+label);
    };
    auto reconstructed_gate = [&](auto pointer, const char* label) {
        const size_t bytes=size_t(tokens)*N*sizeof(float);
        const auto reference=download(*ranks[0],pointer(*ranks[0]),bytes);
        const auto first=download(*ranks[2],pointer(*ranks[2]),bytes);
        const auto second=download(*ranks[3],pointer(*ranks[3]),bytes);
        std::vector<uint8_t> sum(bytes);
        for (size_t j=0; j<bytes; j+=sizeof(float)) {
            float a,b;
            std::memcpy(&a,first.data()+j,sizeof(float));
            std::memcpy(&b,second.data()+j,sizeof(float));
            const float value=a+b;
            std::memcpy(sum.data()+j,&value,sizeof(float));
        }
        numeric_gate(reference,sum,label);
    };
    try {
        sync_all();
        for (int i=0; i<4; ++i) original[i] = std::make_unique<SavedFine>(*ranks[i]);
        for (auto* owner : owners) owner->tokens = tokens;
        std::vector<uint8_t> input_bytes(residual.size()*sizeof(float));
        std::memcpy(input_bytes.data(), residual.data(), input_bytes.size());
        for (auto* rank : ranks) upload(*rank, rank->R, input_bytes);
        // First execute the intact prepared graphs, without proposal/commit
        // publication. These outputs gate the later decomposition independently
        // of its own replay determinism. Restore all storage before proceeding.
        for (int i=0; i<2; ++i)
            ranks[i]->launch(ranks[i]->captured[tokens].mixer[ranks[i]->state_bank]);
        hp.run_hybrid(true);
        sync_all();
        std::array<std::vector<uint8_t>,4> production_attention, production_ffn_input,
            production_shared_hidden, production_routed_hidden, production_shared, production_down;
        for (int i=0; i<4; ++i) {
            auto& rank=*ranks[i];
            production_attention[i]=download(rank,rank.attn_input,size_t(tokens)*N*sizeof(float));
            production_ffn_input[i]=download(rank,rank.ffn_input,size_t(tokens)*N*sizeof(float));
            production_shared_hidden[i]=download(rank,rank.shared_local_q,size_t(tokens)*(rank.ff/32)*36);
            production_routed_hidden[i]=download(rank,rank.hidden(false,tokens),size_t(tokens)*K*(rank.ff/32)*36);
            production_shared[i]=download(rank,rank.shared,size_t(tokens)*N*sizeof(float));
            production_down[i]=download(rank,rank.local_out,size_t(tokens)*N*sizeof(float));
            const auto error=download(rank,rank.plan_error,sizeof(uint32_t));
            uint32_t status=0; std::memcpy(&status,error.data(),sizeof(status));
            if (status) throw std::runtime_error("fine calibration intact graph plan failure");
        }
        for (int i=1; i<4; ++i) if (production_ffn_input[i]!=production_ffn_input[0])
            throw std::runtime_error("fine calibration intact FFN inputs differ");
        exact_canonical([](Rank& rank) { return rank.ids; },size_t(tokens)*K*sizeof(int32_t),"intact route IDs");
        exact_canonical([](Rank& rank) { return rank.weights; },size_t(tokens)*K*sizeof(float),"intact route weights");
        exact_hidden(false);
        exact_hidden(true);
        for (auto& saved : original) saved->restore();
        for (auto* rank : ranks) upload(*rank, rank->R, input_bytes);
        std::vector<uint8_t> canonical_y, canonical_attention, actual_full_down;
        const char* names[] = {"hc-attn", "hc-ffn", "router-plan-quantize", "shared-gu-scalar",
            "routed-gu", "shared-down", "routed-down-combine", "routed-down-combine-fused-control",
            "empty-graph-control"};
        for (int phase=0; phase<9; ++phase) {
            input_hash = 14695981039346656037ull;
            auto& reference = *ranks[0];
            // Materialize only the canonical reference's intervening attention.
            // Other owners receive the exact seam rather than a separately
            // rounded trajectory. Their full-layer gates are the caller's job.
            if (phase == 1) {
                reference.mixer(tokens,true);
                reference.output_projection(tokens);
                reference.sync();
                canonical_y = download(reference, reference.y, size_t(tokens)*V*sizeof(float));
                canonical_attention = download(reference, reference.bo, size_t(tokens)*N*sizeof(float));
            }
            if (phase <= 1) {
                canonical([](Rank& rank) { return rank.R; }, size_t(tokens)*HC*N*sizeof(float));
                if (phase == 1) {
                    canonical([](Rank& rank) { return rank.bo; }, size_t(tokens)*N*sizeof(float));
                    canonical([](Rank& rank) { return rank.inj[0]; }, size_t(tokens)*HC*sizeof(float));
                }
            } else if (phase == 2) {
                canonical([](Rank& rank) { return rank.mixed; }, size_t(tokens)*N*sizeof(float));
            } else if (phase == 3 || phase == 4) {
                canonical([](Rank& rank) { return rank.xq; }, size_t(tokens)*(N/32)*36);
                if (phase == 3)
                    canonical([](Rank& rank) { return rank.mixed; }, size_t(tokens)*N*sizeof(float));
                else {
                    canonical([](Rank& rank) { return rank.ids; }, size_t(tokens)*K*sizeof(int32_t));
                    canonical([](Rank& rank) { return rank.weights; }, size_t(tokens)*K*sizeof(float));
                }
            } else if (phase < 8) {
                sliced_hidden(phase != 5);
                if (phase >= 6) {
                    canonical([](Rank& rank) { return rank.ids; }, size_t(tokens)*K*sizeof(int32_t));
                    canonical([](Rank& rank) { return rank.weights; }, size_t(tokens)*K*sizeof(float));
                    canonical([](Rank& rank) { return rank.scalar; }, size_t(tokens)*sizeof(float));
                    // Column owners retain their valid shared-down partials.
                    // This canonical full shared value identifies the semantic
                    // contribution; it cannot replace a local column partial.
                    hash_bytes(download(reference, reference.shared, size_t(tokens)*N*sizeof(float)));
                }
            }
            auto compute = [&](Rank& rank) {
                select(rank.w.device);
                if (phase == 8) return; // Same event/launch scope, no compute.
                if (phase == 0 || phase == 1) {
                    rank.hc(phase,tokens,phase == 1);
                } else if (phase == 2) {
                    k::bf16_gemv_fp32_mmvf_multi(rank.mixed,N,rank.w.router,rank.logits,NE,N,NE,tokens,rank.compute);
                    k::native_router_top10_multi(rank.logits,rank.ids,rank.weights,tokens,rank.compute);
                    check(cudaMemsetAsync(rank.plan_error,0,sizeof(uint32_t),rank.compute), "fine calibration plan reset");
                    k::resident_plan(rank.ids,tokens*K,K,rank.res,NE,rank.w.expert_arena,rank.offsets,
                        static_cast<long long>(rank.w.expert_bytes),rank.plan,rank.cap_entries,nullptr,0,rank.compute,rank.plan_error);
                    k::native_quantize_q8_1(rank.mixed,rank.xq,N,tokens,rank.compute);
                } else if (phase == 3) {
                    rank.matrix(rank.w.shared_gate,rank.xq,rank.sg,tokens);
                    rank.matrix(rank.w.shared_up,rank.xq,rank.su,tokens);
                    k::native_swiglu_quantize_q8_1(rank.sg,rank.su,rank.shared_local_q,rank.ff,tokens,rank.compute);
                    k::bf16_gemv_fp32_mmvf_multi(rank.mixed,N,rank.w.shared_scalar,rank.scalar,1,N,1,tokens,rank.compute);
                } else if (phase == 4) {
                    rank.grouped(tokens,hp.mode,k::NativeExpertPhase::GateUp,rank.w.gu_layout,rank.gu_scratch);
                } else if (phase == 5) {
                    rank.matrix(rank.w.shared_down,rank.shared_full_q,rank.shared,tokens);
                } else if (rank.ffn_columns() || phase == 7) {
                    const int32_t* starts = rank.plan+4;
                    const int32_t* dst = starts+rank.cap_entries+1;
                    const auto* ptr = reinterpret_cast<const unsigned long long*>(rank.plan+rank.ptr_offset);
                    k::native_expert_down_combine(rank.w.down_layout,ptr,starts,rank.plan,dst,tokens*K,tokens*K,
                        rank.hidden(false,tokens),rank.weights,rank.shared,rank.scalar,rank.local_out,
                        rank.plan_error,tokens,rank.compute);
                } else {
                    rank.grouped(tokens,hp.mode,k::NativeExpertPhase::Down,rank.w.down_layout,rank.down_scratch);
                    k::native_moe_combine_multi_hits_gated(rank.parts,rank.weights,rank.shared,rank.scalar,
                        rank.local_out,rank.width,K,tokens,rank.compute);
                }
            };
            std::array<std::unique_ptr<SavedFine>,4> frozen, produced;
            std::array<std::unique_ptr<FineGraph>,4> graphs;
            for (int i=0; i<4; ++i) frozen[i] = std::make_unique<SavedFine>(*ranks[i]);
            for (auto* rank : ranks) compute(*rank);
            sync_all();
            // The factored phases must also reproduce their unfactored real
            // owner outputs. Hybrid's admitted contract keeps FFN inputs exact.
            for (int i=0; i<4; ++i) {
                auto& rank=*ranks[i];
                const std::vector<uint8_t>* expected=nullptr;
                const void* output=nullptr;
                if (phase==0 || phase==1) {
                    expected=phase==0 ? &production_attention[i] : &production_ffn_input[i];
                    output=rank.mixed;
                } else if (phase==3) { expected=&production_shared_hidden[i]; output=rank.shared_local_q; }
                else if (phase==4) { expected=&production_routed_hidden[i]; output=rank.hidden(false,tokens); }
                else if (phase==5) { expected=&production_shared[i]; output=rank.shared; }
                else if (phase==6 || (phase==7 && i>=2)) { expected=&production_down[i]; output=rank.local_out; }
                if (expected && download(rank,output,expected->size())!=*expected)
                    throw std::runtime_error(std::string("fine calibration intact graph mismatch: ")+names[phase]);
            }
            if (phase==5) reconstructed_gate([](Rank& rank) { return rank.shared; },"shared column reconstruction");
            if (phase==6 || phase==7)
                reconstructed_gate([](Rank& rank) { return rank.local_out; },"FFN column reconstruction");
            if (phase == 0 || phase == 1) {
                exact_canonical([](Rank& rank) { return rank.R; },size_t(tokens)*HC*N*sizeof(float),"HC residual");
                exact_canonical([](Rank& rank) { return rank.mixed; },size_t(tokens)*N*sizeof(float),"HC mixed");
                exact_canonical([&](Rank& rank) { return rank.inj[phase]; },size_t(tokens)*HC*sizeof(float),"HC injection");
            }
            if (phase == 2) {
                exact_canonical([](Rank& rank) { return rank.ids; },size_t(tokens)*K*sizeof(int32_t),"route IDs");
                exact_canonical([](Rank& rank) { return rank.weights; },size_t(tokens)*K*sizeof(float),"route weights");
                exact_canonical([](Rank& rank) { return rank.logits; },size_t(tokens)*NE*sizeof(float),"router logits");
                exact_canonical([](Rank& rank) { return rank.xq; },size_t(tokens)*(N/32)*36,"FFN input Q8");
                // Only active logical plan fields are compared. Resident expert
                // addresses necessarily differ by device and shard allocation.
                const auto bytes = download(reference,reference.plan,sizeof(int32_t));
                std::memcpy(&groups,bytes.data(),sizeof(groups));
                if (groups < 1 || groups > tokens*K) throw std::runtime_error("fine calibration group count");
                const auto start_bytes = download(reference,reference.plan+4,size_t(groups+1)*sizeof(int32_t));
                std::vector<int32_t> starts(groups+1);
                std::memcpy(starts.data(),start_bytes.data(),start_bytes.size());
                group_entries.clear();
                for (int group=0; group<groups; ++group) {
                    if (starts[group+1] <= starts[group]) throw std::runtime_error("fine calibration group entries");
                    group_entries.push_back(starts[group+1]-starts[group]);
                }
                if (starts.front()!=0 || starts.back()!=tokens*K) throw std::runtime_error("fine calibration plan coverage");
                exact_canonical([](Rank& rank) { return rank.plan; },3*sizeof(int32_t),"plan counts");
                exact_canonical([](Rank& rank) { return rank.plan+4; },size_t(groups+1)*sizeof(int32_t),"group starts");
                exact_canonical([](Rank& rank) { return rank.plan+4+rank.cap_entries+1; },size_t(tokens)*K*sizeof(int32_t),"entry destinations");
                exact_canonical([](Rank& rank) { return rank.plan+4+2*rank.cap_entries+1; },size_t(tokens)*K*sizeof(int32_t),"entry tokens");
            }
            if (phase == 3 || phase == 4) {
                exact_hidden(phase == 4);
                if (phase == 3) canonical_shared_hidden=download(reference,reference.shared_local_q,size_t(tokens)*(F/32)*36);
                else canonical_routed_hidden=download(reference,reference.hidden(false,tokens),size_t(tokens)*K*(F/32)*36);
            }
            if (phase == 3)
                exact_canonical([](Rank& rank) { return rank.scalar; },size_t(tokens)*sizeof(float),"shared scalar");
            if (phase == 6) actual_full_down = download(reference,reference.local_out,size_t(tokens)*N*sizeof(float));
            if (phase == 7) {
                for (int index=0; index<2; ++index)
                    numeric_gate(actual_full_down,
                        download(*ranks[index],ranks[index]->local_out,actual_full_down.size()),
                        "full fused control versus original grouped down/combine");
            }
            for (int i=0; i<4; ++i) {
                produced[i] = std::make_unique<SavedFine>(*ranks[i]);
                graphs[i] = std::make_unique<FineGraph>(*ranks[i]);
                ranks[i]->capture(graphs[i]->node,[&,i] { compute(*ranks[i]); });
            }
            const int masks[] = {1,2,4,8,12};
            const char* arms[] = {"full0","full1","half0","half1","concurrent-halves"};
            for (int trial=-warmup; trial<trials; ++trial) for (int order=0; order<10; ++order) {
                // Rotate the five-arm starting point and mirror the second leg.
                // Every arm occurs twice per trial; rank launch order also flips.
                const int rotated=(trial+warmup)%5;
                const int step=order<5 ? order : 9-order;
                const int arm=(step+rotated)%5, mask=masks[arm];
                // Touch each active owner's storage last on its device. Fixed
                // full-then-half restoration would systematically favor halves.
                // This remains an H2D-restore-conditioned, not sustained, probe.
                for (bool active : {false,true}) for (int i=0; i<4; ++i)
                    if (bool(mask&(1<<i))==active) frozen[i]->restore();
                const bool reverse=((trial+warmup+order)&1)!=0;
                const auto start=std::chrono::steady_clock::now();
                for (int j=0; j<4; ++j) {
                    const int i=reverse ? 3-j : j;
                    if (!(mask&(1<<i))) continue;
                    auto& graph=*graphs[i];
                    select(graph.rank.w.device);
                    check(cudaEventRecord(graph.begin,graph.rank.compute),"fine calibration event begin");
                    graph.rank.launch(graph.node);
                    // End is adjacent to its graph, before submitting another
                    // GPU. No cross-device timestamp subtraction is meaningful.
                    check(cudaEventRecord(graph.end,graph.rank.compute),"fine calibration event end");
                }
                sync_all();
                const auto end=std::chrono::steady_clock::now();
                TpGdnFineCalibrationSample sample;
                sample.phase=names[phase]; sample.arm=arms[arm]; sample.owner=arm<2 ? "full" : "hybrid";
                sample.event_scope="adjacent-own-graph-events";
                sample.trial=trial; sample.order=order; sample.restored_bytes=restored_bytes;
                sample.wall_ms=std::chrono::duration<double,std::milli>(end-start).count();
                sample.input_hash=input_hash; sample.groups=groups; sample.group_entries=group_entries;
                for (int i=0; i<4; ++i) {
                    if (mask&(1<<i)) {
                        const int slot=i<2 ? 0 : i-2;
                        select(ranks[i]->w.device);
                        sample.devices[slot]=ranks[i]->w.device;
                        check(cudaEventElapsedTime(&sample.device_ms[slot],graphs[i]->begin,graphs[i]->end),
                              "fine calibration event span");
                        produced[i]->verify();
                    } else frozen[i]->verify();
                    uint32_t error=0;
                    select(ranks[i]->w.device);
                    check(cudaMemcpy(&error,ranks[i]->plan_error,sizeof(error),cudaMemcpyDeviceToHost),"fine calibration plan status");
                    if (error) throw std::runtime_error("fine calibration resident plan failed");
                }
                if (trial>=0) result.push_back(std::move(sample));
            }
            for (auto& saved : produced) saved->restore();
        }
        // Frozen standalone collectives use canonical attention payloads and
        // valid local fused FFN partials. They do not time stale consumers or
        // pretend an isolated publish equals the fused kernel's peer-store cost.
        const std::array<std::vector<uint8_t>,2> ffn_partials{{
            download(*hp.r[0],hp.r[0]->local_out,size_t(tokens)*N*sizeof(float)),
            download(*hp.r[1],hp.r[1]->local_out,size_t(tokens)*N*sizeof(float))}};
        for (int phase=0; phase<3; ++phase) {
            input_hash=14695981039346656037ull;
            for (int rank=0; rank<2; ++rank) {
                auto& a=*hp.r[rank];
                if (phase==0) {
                    const auto mapping=gdn_rank_layout(hp.geometry,rank);
                    std::vector<uint8_t> local(size_t(tokens)*(V/2)*sizeof(float));
                    for (int token=0; token<tokens; ++token) for (int channel=0; channel<V/2; ++channel)
                        std::memcpy(local.data()+(size_t(token)*(V/2)+channel)*sizeof(float),
                            canonical_y.data()+(size_t(token)*V+mapping.value_rows[channel])*sizeof(float),sizeof(float));
                    upload(a,a.y,local);
                } else if (phase==1) {
                    std::vector<uint8_t> local(size_t(tokens)*(N/2)*sizeof(float));
                    for (int token=0; token<tokens; ++token)
                        std::memcpy(local.data()+size_t(token)*(N/2)*sizeof(float),
                            canonical_attention.data()+(size_t(token)*N+rank*(N/2))*sizeof(float),(N/2)*sizeof(float));
                    upload(a,a.local_out,local);
                }
            }
            if (phase==2) for (int rank=0; rank<2; ++rank)
                upload(*hp.r[rank],hp.r[rank]->local_out,ffn_partials[rank]);
            if (phase==0) hash_bytes(canonical_y);
            else if (phase==1) hash_bytes(canonical_attention);
            else for (const auto& partial : ffn_partials) hash_bytes(partial);
            auto destination = [&](Rank& rank) -> float* {
                return phase==0 ? rank.full_y : phase==1 ? rank.bo : rank.peer_partial[1];
            };
            const size_t destination_bytes=size_t(tokens)*(phase==0 ? V : N)*sizeof(float);
            for (int rank=0; rank<2; ++rank) {
                auto& a=*hp.r[rank]; select(a.w.device);
                check(cudaMemsetAsync(destination(a),0xa5,destination_bytes,a.compute),"fine calibration poison exchange");
            }
            sync_all();
            auto push = [&](Rank& rank) {
                select(rank.w.device);
                if (phase==0) hp.push_y(rank);
                else if (phase==1) hp.push_output(rank);
                else k::tp_gdn_publish_partial(rank.local_out,hp.r[1-rank.w.rank]->peer_partial[1],
                                               tokens*N,rank.compute);
            };
            std::array<std::unique_ptr<SavedFine>,4> frozen, produced;
            std::array<std::unique_ptr<FineGraph>,2> pushes, empty;
            for (int i=0; i<4; ++i) frozen[i]=std::make_unique<SavedFine>(*ranks[i]);
            for (int rank=0; rank<2; ++rank) push(*hp.r[rank]);
            hp.phase_barrier(); sync_all();
            for (int rank=0; rank<2; ++rank) {
                const auto& expected=phase==0 ? canonical_y : phase==1 ? canonical_attention : ffn_partials[1-rank];
                if (download(*hp.r[rank],destination(*hp.r[rank]),destination_bytes)!=expected)
                    throw std::runtime_error("fine calibration exchange reconstruction mismatch");
            }
            for (int i=0; i<4; ++i) produced[i]=std::make_unique<SavedFine>(*ranks[i]);
            for (int rank=0; rank<2; ++rank) {
                auto& a=*hp.r[rank];
                pushes[rank]=std::make_unique<FineGraph>(a);
                empty[rank]=std::make_unique<FineGraph>(a);
                a.capture(pushes[rank]->node,[&,rank] { push(*hp.r[rank]); });
                a.capture(empty[rank]->node,[] {});
            }
            const char* names[]={"attn-y-exchange","attn-output-exchange","ffn-partial-exchange"};
            const uint64_t payload_per_token[]={12288,5120,10240};
            for (int trial=-warmup; trial<trials; ++trial) for (int order=0; order<4; ++order) {
                const bool exchange=(order==1 || order==2)^bool(trial&1);
                for (auto& saved : frozen) saved->restore();
                auto& graphs=exchange ? pushes : empty;
                const bool reverse=((trial+warmup+order)&1)!=0;
                const auto start=std::chrono::steady_clock::now();
                for (int j=0; j<2; ++j) {
                    const int rank=reverse ? 1-j : j;
                    auto& graph=*graphs[rank]; select(graph.rank.w.device);
                    check(cudaEventRecord(graph.begin,graph.rank.compute),"fine calibration exchange begin");
                    graph.rank.launch(graph.node);
                }
                hp.phase_barrier();
                for (int j=0; j<2; ++j) {
                    const int rank=reverse ? 1-j : j;
                    auto& graph=*graphs[rank]; select(graph.rank.w.device);
                    check(cudaEventRecord(graph.end,graph.rank.compute),"fine calibration exchange joined end");
                }
                sync_all();
                const auto end=std::chrono::steady_clock::now();
                TpGdnFineCalibrationSample sample;
                sample.phase=names[phase]; sample.arm=exchange ? "exchange" : "event-join-empty";
                sample.owner="hybrid"; sample.event_scope="graph-plus-peer-event-join";
                sample.trial=trial; sample.order=order; sample.restored_bytes=restored_bytes;
                sample.peer_bytes_per_rank=exchange ? payload_per_token[phase]*tokens : 0;
                sample.input_hash=input_hash; sample.groups=groups; sample.group_entries=group_entries;
                sample.wall_ms=std::chrono::duration<double,std::milli>(end-start).count();
                for (int rank=0; rank<2; ++rank) {
                    select(hp.r[rank]->w.device); sample.devices[rank]=hp.r[rank]->w.device;
                    check(cudaEventElapsedTime(&sample.device_ms[rank],graphs[rank]->begin,graphs[rank]->end),
                          "fine calibration exchange event span");
                }
                for (int i=0; i<4; ++i) (exchange ? produced[i] : frozen[i])->verify();
                if (trial>=0) result.push_back(std::move(sample));
            }
        }
        // Give every phase the route geometry from this canonical T fixture,
        // including HC phases that execute before routing is materialized.
        for (auto& sample : result) { sample.groups=groups; sample.group_entries=group_entries; }
        for (auto& saved : original) saved->restore();
        for (const auto& saved : original) saved->verify();
        for (size_t i=0; i<owners.size(); ++i) owners[i]->tokens=old_tokens[i];
        return result;
    } catch (...) {
        for (size_t i=0; i<owners.size(); ++i) {
            owners[i]->tokens=old_tokens[i];
            owners[i]->poison_and_drain();
        }
        // Preserve caller storage when the runtime remains usable. The owners
        // still stay poisoned, so a failed probe never looks like a valid run.
        try { for (auto& saved : original) if (saved) saved->restore(); } catch (...) {}
        throw;
    }
}

void TpGdnLayer::prepare_flat(int tokens,bool shard_hc,bool profile,uint64_t timeout_us) {
    auto& p=*impl_;p.healthy();
    if(p.ffn_columns())throw std::invalid_argument("TP GDN column partition does not use the flat protocol");
    if(tokens<1||tokens>p.capacity||timeout_us<1||timeout_us>10000000)
        throw std::invalid_argument("TP GDN flat preparation requires T within capacity and timeout 1us..10s");
    if(p.flat_prepared[tokens][shard_hc][profile]){
        if(p.flat_timeout[tokens][shard_hc][profile]!=timeout_us)throw std::logic_error("TP GDN captured timeout cannot be changed");
        return;
    }
    if(p.seen_epoch||p.pending)throw std::logic_error("TP GDN flat preparation is startup-only");
    prepare_captured(tokens); // initialize dispatch and provide bank-specific commit graphs
    const int old_tokens=p.tokens;
    try{p.setup_flat();p.tokens=tokens;
        for(int i=0;i<p.count;++i){auto& a=*p.r[i];const int original=a.state_bank;a.trace=profile;
            try{for(int bank=0;bank<2;++bank){if(a.state_bank!=bank)a.swap_state_bank();
                a.capture(a.captured[tokens].flat[shard_hc][profile][bank],[&]{p.flat_graph(a,shard_hc,timeout_us);});
            }}catch(...){if(a.state_bank!=original)a.swap_state_bank();a.trace=false;throw;}
            if(a.state_bank!=original)a.swap_state_bank();
            a.trace=false;
        }
        p.flat_prepared[tokens][shard_hc][profile]=true;p.flat_timeout[tokens][shard_hc][profile]=timeout_us;p.tokens=old_tokens;
    }catch(...){p.abort_flat();p.tokens=old_tokens;p.poisoned=true;throw;}
}
void TpGdnLayer::set_execution(TpGdnExecution mode,bool profile) {
    auto& p=*impl_;p.healthy();
    if(p.pending)throw std::logic_error("TP GDN execution mode cannot change during an outstanding proposal");
    if(mode!=TpGdnExecution::RuntimeCopies&&mode!=TpGdnExecution::Consolidated&&mode!=TpGdnExecution::Captured&&mode!=TpGdnExecution::ColumnCaptured&&mode!=TpGdnExecution::HybridCaptured&&mode!=TpGdnExecution::HybridRcclCaptured&&mode!=TpGdnExecution::FlatCaptured&&mode!=TpGdnExecution::FlatHcCaptured)
        throw std::invalid_argument("TP GDN invalid execution mode");
    if(p.projection_columns()&&mode!=TpGdnExecution::ColumnCaptured)
        throw std::invalid_argument("TP GDN input-column owners require ColumnCaptured execution");
    if(p.count==2&&!p.projection_columns()&&mode==TpGdnExecution::ColumnCaptured)
        throw std::invalid_argument("TP GDN ColumnCaptured requires input-column weights");
    if(p.hybrid()&&mode!=TpGdnExecution::HybridCaptured&&mode!=TpGdnExecution::HybridRcclCaptured)
        throw std::invalid_argument("TP GDN hybrid owners require a hybrid execution mode");
    if(p.count==2&&!p.hybrid()&&mode==TpGdnExecution::HybridCaptured)
        throw std::invalid_argument("TP GDN HybridCaptured requires hybrid weights");
    if(mode==TpGdnExecution::HybridRcclCaptured&&(!p.hybrid()||profile))
        throw std::invalid_argument("TP GDN RCCL requires unprofiled hybrid weights");
#if !defined(STRATA_TP2_GDN_RCCL)
    if(mode==TpGdnExecution::HybridRcclCaptured)throw std::runtime_error("TP GDN RCCL was not built");
#endif
    if(profile&&mode!=TpGdnExecution::Captured&&mode!=TpGdnExecution::ColumnCaptured&&mode!=TpGdnExecution::HybridCaptured&&mode!=TpGdnExecution::FlatCaptured&&mode!=TpGdnExecution::FlatHcCaptured)
        throw std::invalid_argument("TP GDN GPU profiling requires a captured graph variant");
    p.execution=mode;p.profile_enabled=profile;p.last_profiled=false;
}
TpGdnExecution TpGdnLayer::execution() const {return impl_->execution;}
void TpGdnLayer::set_flat_fault_for_test(int missing,int delayed,uint64_t delay) {
    auto& p=*impl_;p.healthy();
    if(p.pending||p.count!=2||!p.flat_mode()||missing < -1||missing>1||delayed < -1||delayed>1||delay>10000000)
        throw std::invalid_argument("TP GDN fault injection requires idle two-rank flat mode and valid rank/delay");
    if(missing>=0&&delayed>=0)throw std::invalid_argument("TP GDN choose missing OR delayed peer fault");
    p.missing_rank=missing;p.delayed_rank=delayed;p.delay_us=delay;
}
TpGdnLayerProfile TpGdnLayer::profile(int rank) const {
    const auto& p=*impl_;p.healthy();
    if(rank<0||rank>=p.count||!p.last_profiled)throw std::logic_error("TP GDN has no successful profiled proposal for that rank");
    p.sync();auto& a=*p.r[rank];select(a.w.device);
    static const char* labels[]={"begin","hc_attn_done","qkv_z_done","conv_done","ab_done","gdn_done",
        "y_push_done","y_wait_done","y_unpack_done","output_projection_done","attn_push_done","attn_wait_done","attn_unpack_done",
        "hc_ffn_done","router_done","shared_gu_done","routed_gu_done","hidden_push_done","hidden_wait_done","hidden_unpack_done",
        "shared_down_done","routed_down_done","combine_done","ffn_push_done","ffn_wait_done","ffn_unpack_done","hc_write_done","end",
        "hc0_down_done","hc0_lo_push_done","hc0_lo_wait_done","hc0_lo_unpack_done","hc0_up_done","hc0_mix_push_done","hc0_mix_wait_done","hc0_mix_unpack_done",
        "hc1_down_done","hc1_lo_push_done","hc1_lo_wait_done","hc1_lo_unpack_done","hc1_up_done","hc1_mix_push_done","hc1_mix_wait_done","hc1_mix_unpack_done"};
    TpGdnLayerProfile out;out.epoch=p.epoch;out.tokens=p.tokens;out.rank=rank;
    out.labels.assign(std::begin(labels),std::end(labels));out.nanoseconds.resize(44);
    if(!p.flat_mode()){
        // Inter-segment intervals include event/copy ordering and host arrival
        // skew; they are not isolated transfer latency. No cross-GPU subtraction.
        out.labels[6]="y_producer_end";out.labels[8]="y_consumer_begin";
        out.labels[10]="attn_producer_end";out.labels[12]="attn_consumer_begin";
        out.labels[17]="hidden_producer_end";out.labels[19]="hidden_consumer_begin";
        out.labels[23]="ffn_producer_end";out.labels[25]="ffn_consumer_begin";
        if(p.projection_columns()){
            out.labels[11]="attn_reduce_begin";out.labels[12]="attn_reduce_done";
        }
        if(p.ffn_columns()){
            out.labels[21]="routed_down_combine_done";
            out.labels[24]="ffn_reduce_begin";out.labels[25]="ffn_reduce_done";
        }
    }
    check(cudaMemcpy(out.nanoseconds.data(),a.stamps,44*sizeof(uint64_t),cudaMemcpyDeviceToHost),"GPU phase stamps");
    if(p.count==2&&p.flat_mode()){k::TpGdnProtocolStamp status[8];check(cudaMemcpy(status,a.protocol_stamps,sizeof(status),cudaMemcpyDeviceToHost),"GPU wait stamps");
        for(int i=0;i<8;++i){out.wait_nanoseconds[i]=status[i].elapsed_ticks*1000000/a.wall_khz;out.wait_polls[i]=status[i].polls;out.wait_status[i]=status[i].status;}
    }
    return out;
}
void TpGdnLayer::reset_state(const std::vector<float>& state,const std::vector<float>& conv) {
    auto& p=*impl_;p.healthy();
    if(state.size()!=128u*48u*128u||conv.size()!=3u*C)throw std::invalid_argument("TP GDN canonical state shape mismatch");
    try {p.sync();for(int i=0;i<p.count;++i){auto& r=*p.r[i];select(r.w.device);
        std::vector<float> st(128*r.hv*128),cv(3*r.channels);
        if(p.count==1){st=state;cv=conv;}else{const auto map=gdn_rank_layout(p.geometry,i);
            for(int row=0;row<128;++row)for(int h=0;h<24;++h)for(int col=0;col<128;++col)
                st[map.local_state_index(row,h,col)]=state[map.global_state_index(row,h,col)];
            for(int tap=0;tap<3;++tap)for(int c=0;c<r.channels;++c)cv[c*3+tap]=conv[map.qkv_rows[c]*3+tap];}
        check(cudaMemcpy(r.state,st.data(),st.size()*sizeof(float),cudaMemcpyHostToDevice),"state reset");
        check(cudaMemcpy(r.conv,cv.data(),cv.size()*sizeof(float),cudaMemcpyHostToDevice),"conv reset");}
        p.pending=false;
    }catch(...){p.poisoned=true;throw;}
}
void TpGdnLayer::propose(const std::vector<float>& residual,int tokens,uint64_t epoch) {
    auto& p=*impl_;if(residual.size()!=static_cast<size_t>(tokens)*HC*N)throw std::invalid_argument("TP GDN residual shape mismatch");
    p.begin(tokens,epoch);
    try{
#if defined(STRATA_TP2_GDN_RCCL)
        if(p.rccl_mode()){p.run_rccl([&](Rank& r){check(cudaMemcpyAsync(r.R,residual.data(),residual.size()*sizeof(float),cudaMemcpyHostToDevice,r.compute),"RCCL residual upload");});return;}
#endif
        for(int i=0;i<p.count;++i){auto& r=*p.r[i];select(r.w.device);check(cudaMemcpyAsync(r.R,residual.data(),residual.size()*sizeof(float),cudaMemcpyHostToDevice,r.compute),"residual upload");}p.run();}
    catch(...){p.poison_and_drain();throw;}
}
void TpGdnLayer::propose_device(const std::array<const float*,2>& residual,int tokens,uint64_t epoch) {
    auto& p=*impl_;for(int i=0;i<p.count;++i)if(!residual[i])throw std::invalid_argument("TP GDN missing rank residual");
    p.begin(tokens,epoch);
    try{
#if defined(STRATA_TP2_GDN_RCCL)
        if(p.rccl_mode()){p.run_rccl([&](Rank& r){check(cudaMemcpyAsync(r.R,residual[r.w.rank],tokens*HC*N*sizeof(float),cudaMemcpyDeviceToDevice,r.compute),"RCCL residual device copy");});return;}
#endif
        for(int i=0;i<p.count;++i){auto& r=*p.r[i];select(r.w.device);check(cudaMemcpyAsync(r.R,residual[i],tokens*HC*N*sizeof(float),cudaMemcpyDeviceToDevice,r.compute),"residual device copy");}p.run();}
    catch(...){p.poison_and_drain();throw;}
}
void TpGdnLayer::commit(int keep) {
    auto& p=*impl_;p.healthy();if(!p.pending||keep<0||keep>p.tokens)throw std::invalid_argument("TP GDN commit requires an outstanding window and keep in 0..T");
    if(keep==0){p.pending=false;return;}
    try{
#if defined(STRATA_TP2_GDN_RCCL)
        if(p.rccl_mode()){
            auto command=[&](int rank){auto& r=*p.r[rank];select(r.w.device);
                check(cudaMemcpyAsync(r.keep,&keep,sizeof(keep),cudaMemcpyHostToDevice,r.compute),"RCCL commit count");
                r.launch(r.captured[p.tokens].commit[r.state_bank]);p.rccl->complete(rank);};
            p.rccl->paired("layer-commit",std::ref(command));
        }
        else
#endif
        {for(int i=0;i<p.count;++i){auto& r=*p.r[i];select(r.w.device);
        check(cudaMemcpyAsync(r.keep,&keep,sizeof(keep),cudaMemcpyHostToDevice,r.compute),"commit count");
        if(p.execution==TpGdnExecution::Captured||p.execution==TpGdnExecution::ColumnCaptured||p.execution==TpGdnExecution::HybridCaptured||p.flat_mode())r.launch(r.captured[p.tokens].commit[r.state_bank]);
        else r.stage_commit(p.tokens);
    }p.sync();}
    // Pointer publication is all-host, after both successful device completions.
    for(int i=0;i<p.count;++i)p.r[i]->swap_state_bank();
    p.pending=false;
    }catch(...){
        p.poisoned=true;
#if defined(STRATA_TP2_GDN_RCCL)
        if(p.rccl)p.rccl->fail("layer commit failed; no bank publication is valid");
#endif
        throw;
    }
}
const float* TpGdnLayer::residual(int rank) const {impl_->healthy();if(rank<0||rank>=impl_->count)throw std::out_of_range("TP GDN rank");return impl_->r[rank]->R;}
int TpGdnLayer::device(int rank) const {if(rank<0||rank>=impl_->count)throw std::out_of_range("TP GDN rank");return impl_->r[rank]->w.device;}
int TpGdnLayer::ranks() const {return impl_->count;}

TpGdnLayerSnapshot TpGdnLayer::snapshot(int rank) const {
    const auto& p=*impl_;p.healthy();if(rank<0||rank>=p.count)throw std::out_of_range("TP GDN rank");p.sync();
    auto& r=*p.r[rank];select(r.w.device);TpGdnLayerSnapshot s;s.tokens=p.tokens;s.epoch=p.epoch;
    auto get=[]<class T>(std::vector<T>& out,const T* device,size_t n){out.resize(n);if(n)check(cudaMemcpy(out.data(),device,n*sizeof(T),cudaMemcpyDeviceToHost),"diagnostic read");};
    get(s.residual,r.R,p.tokens*HC*N);get(s.attention_input,r.attn_input,p.tokens*N);get(s.ffn_input,r.ffn_input,p.tokens*N);
    if(!p.projection_columns())get(s.gdn_output,r.full_y,p.tokens*V);
    else{
        s.gdn_output.resize(p.tokens*V);
        for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);std::vector<float> local;
            get(local,a.y,p.tokens*a.value);const auto map=gdn_rank_layout(p.geometry,i);
            for(int t=0;t<p.tokens;++t)for(int c=0;c<a.value;++c)s.gdn_output[t*V+map.value_rows[c]]=local[t*a.value+c];
        }
        select(r.w.device);
    }
    get(s.output,r.bo,p.tokens*N);get(s.ids,r.ids,p.tokens*K);get(s.route_weights,r.weights,p.tokens*K);
    get(s.mixer_output,r.mixer_output,p.tokens*N);get(s.route_logits,r.logits,p.tokens*NE);
    get(s.shared_gate_logits,r.scalar,p.tokens);
    s.shared_output.resize(p.tokens*N);s.expert_parts.resize(p.tokens*K*N);
    if(p.ffn_columns()){
        // Fused inference never materializes expert parts. Recompute only for
        // diagnostics, from the retained grouped hidden rows and resident plan.
        for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);
            a.grouped(p.tokens,p.mode,k::NativeExpertPhase::Down,a.w.down_layout,a.down_scratch);}
        p.sync();select(r.w.device);
        get(s.local_output_partial,r.local_out,p.tokens*N);
        get(s.local_shared_partial,r.shared,p.tokens*N);
        get(s.local_expert_parts,r.parts,p.tokens*K*N);
    }
    for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);std::vector<float> sh,part;get(sh,a.shared,p.tokens*a.width);get(part,a.parts,p.tokens*K*a.width);
        if(p.ffn_columns()){
            if(i==0){s.shared_output=sh;s.expert_parts=part;}
            else{for(size_t j=0;j<sh.size();++j)s.shared_output[j]+=sh[j];
                for(size_t j=0;j<part.size();++j)s.expert_parts[j]+=part[j];}
        }else{
            const int offset=p.count==1?0:i*a.width;
            for(int t=0;t<p.tokens;++t)std::copy_n(sh.data()+t*a.width,a.width,s.shared_output.data()+t*N+offset);
            for(int e=0;e<p.tokens*K;++e)std::copy_n(part.data()+e*a.width,a.width,s.expert_parts.data()+e*N+offset);
        }
    }
    select(r.w.device);
    if(!p.ffn_columns()){
        get(s.routed_hidden_q8,r.hidden(true,p.tokens),static_cast<size_t>(p.tokens)*K*(F/32)*36);
        get(s.shared_hidden_q8,r.shared_full_q,static_cast<size_t>(p.tokens)*(F/32)*36);
    }else{
        constexpr size_t half=(F/2/32)*36,full=2*half;
        s.routed_hidden_q8.resize(static_cast<size_t>(p.tokens)*K*full);
        s.shared_hidden_q8.resize(static_cast<size_t>(p.tokens)*full);
        for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);std::vector<uint8_t> hidden,sh;
            get(hidden,a.hidden(false,p.tokens),static_cast<size_t>(p.tokens)*K*half);
            get(sh,a.shared_local_q,static_cast<size_t>(p.tokens)*half);
            for(int e=0;e<p.tokens*K;++e)std::copy_n(hidden.data()+e*half,half,s.routed_hidden_q8.data()+e*full+i*half);
            for(int t=0;t<p.tokens;++t)std::copy_n(sh.data()+t*half,half,s.shared_hidden_q8.data()+t*full+i*half);
        }
    }
    s.state.resize(128u*48u*128u);s.conv.resize(3u*C);
    for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);std::vector<float> st,cv;get(st,a.state,128*a.hv*128);get(cv,a.conv,3*a.channels);
        if(p.count==1){s.state=std::move(st);s.conv=std::move(cv);}else{const auto map=gdn_rank_layout(p.geometry,i);
            for(int row=0;row<128;++row)for(int h=0;h<24;++h)for(int col=0;col<128;++col)s.state[map.global_state_index(row,h,col)]=st[map.local_state_index(row,h,col)];
            for(int tap=0;tap<3;++tap)for(int c=0;c<a.channels;++c)s.conv[map.qkv_rows[c]*3+tap]=cv[c*3+tap];}
    }
    return s;
}
} // namespace strata::core::tp2
