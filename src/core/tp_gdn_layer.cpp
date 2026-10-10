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
};
struct Rank {
    TpGdnRankWeights w;
    int capacity, hk, hv, channels, value, ff, width;
    cudaStream_t compute{},copy{};
    cudaEvent_t ready{},arrival{};
    std::vector<void*> owned;
    std::array<CapturedWindow,9> captured{}, captured_profile{};
    bool columns() const {return w.rank>=0 && w.partition==TpGdnPartition::InputColumns;}
    CapturedWindow& window(int t,bool profile){return profile?captured_profile[t]:captured[t];}
    int state_bank=0;
    bool trace=false;
    unsigned long long* stamps{};
    float *hc_local_lo{},*hc_local_mixed{};
    std::array<uint8_t*,8> inbox{};
    k::TpGdnProtocolStamp* protocol_stamps{};
    k::TpGdnProtocolControl* control{};
    uint64_t wall_khz=0;
    float *R{},*mixed{},*attn_input{},*ffn_input{},*mixer_output{},*lo{},*rs{},*xn{},*inj[2]{},*bo{},*local_out{};
    float* peer_partial[2]{}; // distinct attention/FFN receive payloads
    float *state{},*conv{},*candidate_state{},*candidate_conv{},*qkv{},*h{},*gate{},*beta{},*z{},*y{},*full_y{};
    float *logits{},*weights{},*sg{},*su{},*scalar{},*shared{},*parts{};
    uint8_t *xq{},*shared_local_q{},*shared_full_q{},*gu_scratch{},*down_scratch{};
    int32_t *ids{},*res{},*plan{},*keep{};
    unsigned long long* offsets{};
    uint32_t* plan_error{};
    int cap_entries, ptr_offset;
    explicit Rank(const TpGdnRankWeights& weights_in,int cap):w(weights_in),capacity(cap),
        hk(w.rank<0?16:8),hv(w.rank<0?48:24),channels(w.rank<0?C:C/2),
        value(w.rank<0?V:V/2),ff(w.rank<0?F:F/2),width(w.rank<0||w.partition==TpGdnPartition::InputColumns?N:N/2),cap_entries(cap*K) {
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
            if(columns())for(auto& p:peer_partial)p=alloc<float>(cap*N);
            state=alloc<float>(128*hv*128);conv=alloc<float>(3*channels);
            candidate_state=alloc<float>(128*hv*128);candidate_conv=alloc<float>(3*channels);
            qkv=alloc<float>(cap*channels);h=alloc<float>(cap*channels);gate=alloc<float>(cap*hv);beta=alloc<float>(cap*hv);
            z=alloc<float>(cap*value);y=alloc<float>(cap*value);full_y=w.rank<0||columns()?y:alloc<float>(cap*V);
            logits=alloc<float>(cap*NE);weights=alloc<float>(cap*K);ids=alloc<int32_t>(cap*K);
            sg=alloc<float>(cap*ff);su=alloc<float>(cap*ff);scalar=alloc<float>(cap);shared=alloc<float>(cap*width);
            parts=alloc<float>(cap*K*width);xq=alloc<uint8_t>(cap*(V/32)*36);
            shared_local_q=alloc<uint8_t>(cap*(ff/32)*36);shared_full_q=w.rank<0||columns()?shared_local_q:alloc<uint8_t>(cap*(F/32)*36);
            gu_scratch=alloc<uint8_t>(k::native_expert_scratch_bytes(cap_entries,ff));
            // Full reference and input-column owners consume their own GU
            // hidden in down. Only output-row TP needs a full-width gather.
            down_scratch=w.rank<0||columns()?gu_scratch:alloc<uint8_t>(k::native_expert_scratch_bytes(cap_entries,F));
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
        void* p=nullptr;check(cudaMalloc(&p,n*sizeof(T)),"allocation");owned.push_back(p);
        check(cudaMemset(p,0,n*sizeof(T)),"initial zero");return static_cast<T*>(p);
    }
    void release() noexcept {
        cudaSetDevice(w.device);
        if(compute)cudaStreamSynchronize(compute);
        if(copy)cudaStreamSynchronize(copy);
        // Graph nodes borrow weight/state/scratch pointers: destroy after drain,
        // on their owning device, BEFORE freeing any referenced allocations.
        auto destroy=[](CapturedNode& node){
            if(node.executable)cudaGraphExecDestroy(node.executable);
            if(node.graph)cudaGraphDestroy(node.graph);
            node={};
        };
        for(auto* windows:{&captured,&captured_profile})for(auto& window:*windows){
            for(auto& node:window.mixer)destroy(node);
            for(auto& node:window.middle)destroy(node);
            for(auto& node:window.commit)destroy(node);
            for(auto& hc:window.flat)for(auto& profile:hc)for(auto& node:profile)destroy(node);
        }
        for(void* p:owned)cudaFree(p);
        owned.clear();
        if(ready)cudaEventDestroy(ready);
        if(arrival)cudaEventDestroy(arrival);
        if(compute)cudaStreamDestroy(compute);
        if(copy)cudaStreamDestroy(copy);
        ready={};arrival={};compute={};copy={};
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
        return (full?down_scratch:gu_scratch)+k::native_expert_hidden_q8_offset(t*K,full&&!columns()?F:ff);
    }
    void ffn_down(int t,int mode,float* peer_output=nullptr) {
        select(w.device);matrix(w.shared_down,shared_full_q,shared,t);stamp(20);
        if(columns()){
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
    bool columns() const {return count==2 && r[0]->columns();}
    bool flat_prepared[9][2][2]{};
    uint64_t flat_timeout[9][2][2]{};
    k::TpGdnProtocolControl* host_control{};
    uint64_t protocol_epoch=0;
    bool profile_enabled=false,last_profiled=false;
    int missing_rank=-1,delayed_rank=-1;
    uint64_t delay_us=0;
    bool flat_mode() const {return execution==TpGdnExecution::FlatCaptured||execution==TpGdnExecution::FlatHcCaptured;}
    void abort_flat() noexcept {
        if(host_control)__atomic_store_n(&host_control->host_abort.value,uint64_t{1},__ATOMIC_RELEASE);
    }
    void poison_and_drain() noexcept {
        abort_flat();poisoned=true;
        for(auto& rank:r)if(rank){cudaSetDevice(rank->w.device);cudaStreamSynchronize(rank->compute);cudaStreamSynchronize(rank->copy);}
    }
    ~Impl(){
        abort_flat(); // release any bounded waits before owning streams drain
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
            const bool full=w.rank<0,col=!full&&w.partition==TpGdnPartition::InputColumns;
            if(w.partition!=TpGdnPartition::OutputRows&&w.partition!=TpGdnPartition::InputColumns)
                throw std::invalid_argument("TP GDN unknown weight partition");
            const int c=full?C:C/2,v=full?V:V/2,f=full?F:F/2,n=full||col?N:N/2,df=col?F/2:F;
            auto matrix=[](const TpNativeMatrix& x,int in,int out){if(!x.data||!k::native_mmvq_supported(x.type)||x.input!=in||x.output!=out)throw std::invalid_argument("TP GDN invalid native matrix descriptor");
                (void)k::native_mmvq_weight_bytes(x.type,in,out);};
            matrix(w.qkv,N,c);matrix(w.z,N,v);matrix(w.out,col?V/2:V,n);matrix(w.shared_gate,N,f);matrix(w.shared_up,N,f);matrix(w.shared_down,df,n);
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
        if(columns())execution=TpGdnExecution::ColumnCaptured;
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
                a.owned.push_back(ptr);a.inbox[phase]=static_cast<uint8_t*>(ptr);
                check(cudaMemset(ptr,0,capacity*per_token[phase]),"initialize peer inbox");
            }
            check(cudaDeviceSynchronize(),"flat initialization fence");
        }
#else
        throw std::runtime_error("TP GDN flat protocol requires the gfx906 HIP build");
#endif
    }
    void healthy() const {if(poisoned)throw std::runtime_error("TP GDN session poisoned by prior device failure");}
    void sync() const {for(int i=0;i<count;++i)r[i]->sync();}
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
        if(columns())run_column(execution==TpGdnExecution::ColumnCaptured);
        else if(flat_mode())run_flat();
        else if(execution==TpGdnExecution::RuntimeCopies)run_runtime_copies();
        else run_segmented(execution==TpGdnExecution::Captured||execution==TpGdnExecution::ColumnCaptured);
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
        if((execution==TpGdnExecution::Captured||execution==TpGdnExecution::ColumnCaptured)&&!(profile_enabled?prepared_profile[t]:prepared[t]))throw std::logic_error("TP GDN token count was not captured during startup");
        if(flat_mode()&&!flat_prepared[t][execution==TpGdnExecution::FlatHcCaptured][profile_enabled])
            throw std::logic_error("TP GDN flat token/profile/HC variant was not prepared during startup");
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
        // candidate state. Column owners execute their two-reduction DAG here.
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
                        else if(p.columns())p.column_phase(a,0);
                        else p.phase(a,0);
                    });
                    if(!profile)a.capture(window.commit[bank],[&]{a.stage_commit(tokens);});
                }
                if(a.state_bank!=original_bank)a.swap_state_bank();
                if(p.count==2)for(int ph=1;ph<(p.columns()?3:5);++ph)
                    a.capture(window.middle[ph-1],[&]{if(p.columns())p.column_phase(a,ph);else p.phase(a,ph);});
                a.trace=false;
            }catch(...){if(a.state_bank!=original_bank)a.swap_state_bank();a.trace=false;throw;}
        }
        (profile?p.prepared_profile[tokens]:p.prepared[tokens])=true;
        p.execution=old_execution;p.tokens=old_tokens;p.profile_enabled=old_profile;
    }catch(...){p.execution=old_execution;p.tokens=old_tokens;p.profile_enabled=old_profile;p.pending=false;p.poisoned=true;throw;}
}
void TpGdnLayer::prepare_flat(int tokens,bool shard_hc,bool profile,uint64_t timeout_us) {
    auto& p=*impl_;p.healthy();
    if(p.columns())throw std::invalid_argument("TP GDN column partition does not use the flat protocol");
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
    if(mode!=TpGdnExecution::RuntimeCopies&&mode!=TpGdnExecution::Consolidated&&mode!=TpGdnExecution::Captured&&mode!=TpGdnExecution::ColumnCaptured&&mode!=TpGdnExecution::FlatCaptured&&mode!=TpGdnExecution::FlatHcCaptured)
        throw std::invalid_argument("TP GDN invalid execution mode");
    if(p.columns()&&mode!=TpGdnExecution::ColumnCaptured)
        throw std::invalid_argument("TP GDN input-column owners require ColumnCaptured execution");
    if(p.count==2&&!p.columns()&&mode==TpGdnExecution::ColumnCaptured)
        throw std::invalid_argument("TP GDN ColumnCaptured requires input-column weights");
    if(profile&&mode!=TpGdnExecution::Captured&&mode!=TpGdnExecution::ColumnCaptured&&mode!=TpGdnExecution::FlatCaptured&&mode!=TpGdnExecution::FlatHcCaptured)
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
        if(p.columns()){
            out.labels[11]="attn_reduce_begin";out.labels[12]="attn_reduce_done";
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
    try{for(int i=0;i<p.count;++i){auto& r=*p.r[i];select(r.w.device);check(cudaMemcpyAsync(r.R,residual.data(),residual.size()*sizeof(float),cudaMemcpyHostToDevice,r.compute),"residual upload");}p.run();}
    catch(...){p.poison_and_drain();throw;}
}
void TpGdnLayer::propose_device(const std::array<const float*,2>& residual,int tokens,uint64_t epoch) {
    auto& p=*impl_;for(int i=0;i<p.count;++i)if(!residual[i])throw std::invalid_argument("TP GDN missing rank residual");
    p.begin(tokens,epoch);
    try{for(int i=0;i<p.count;++i){auto& r=*p.r[i];select(r.w.device);check(cudaMemcpyAsync(r.R,residual[i],tokens*HC*N*sizeof(float),cudaMemcpyDeviceToDevice,r.compute),"residual device copy");}p.run();}
    catch(...){p.poison_and_drain();throw;}
}
void TpGdnLayer::commit(int keep) {
    auto& p=*impl_;p.healthy();if(!p.pending||keep<0||keep>p.tokens)throw std::invalid_argument("TP GDN commit requires an outstanding window and keep in 0..T");
    if(keep==0){p.pending=false;return;}
    try{for(int i=0;i<p.count;++i){auto& r=*p.r[i];select(r.w.device);
        check(cudaMemcpyAsync(r.keep,&keep,sizeof(keep),cudaMemcpyHostToDevice,r.compute),"commit count");
        if(p.execution==TpGdnExecution::Captured||p.execution==TpGdnExecution::ColumnCaptured||p.flat_mode())r.launch(r.captured[p.tokens].commit[r.state_bank]);
        else r.stage_commit(p.tokens);
    }p.sync();
    // Pointer publication is all-host, after both successful device completions.
    for(int i=0;i<p.count;++i)p.r[i]->swap_state_bank();
    p.pending=false;
    }catch(...){p.poisoned=true;throw;}
}
const float* TpGdnLayer::residual(int rank) const {impl_->healthy();if(rank<0||rank>=impl_->count)throw std::out_of_range("TP GDN rank");return impl_->r[rank]->R;}
int TpGdnLayer::device(int rank) const {if(rank<0||rank>=impl_->count)throw std::out_of_range("TP GDN rank");return impl_->r[rank]->w.device;}
int TpGdnLayer::ranks() const {return impl_->count;}

TpGdnLayerSnapshot TpGdnLayer::snapshot(int rank) const {
    const auto& p=*impl_;p.healthy();if(rank<0||rank>=p.count)throw std::out_of_range("TP GDN rank");p.sync();
    auto& r=*p.r[rank];select(r.w.device);TpGdnLayerSnapshot s;s.tokens=p.tokens;s.epoch=p.epoch;
    auto get=[]<class T>(std::vector<T>& out,const T* device,size_t n){out.resize(n);if(n)check(cudaMemcpy(out.data(),device,n*sizeof(T),cudaMemcpyDeviceToHost),"diagnostic read");};
    get(s.residual,r.R,p.tokens*HC*N);get(s.attention_input,r.attn_input,p.tokens*N);get(s.ffn_input,r.ffn_input,p.tokens*N);
    if(!p.columns())get(s.gdn_output,r.full_y,p.tokens*V);
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
    if(p.columns()){
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
        if(p.columns()){
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
    if(!p.columns()){
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
