#include "strata/core/tp_gdn_layer.hpp"
#include "strata/core/tp_layer_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::core::tp2 {
namespace {
namespace k = strata::kernels;
constexpr int N=2560, F=640, HC=4, K=10, NE=512, C=10240, V=6144;
constexpr float eps=1e-6f;
void check(cudaError_t e, const char* what) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string("TP GDN ")+what+": "+cudaGetErrorString(e));
}
void select(int device) { check(cudaSetDevice(device),"select device"); }
struct Rank {
    TpGdnRankWeights w;
    int capacity, hk, hv, channels, value, ff, width;
    cudaStream_t compute{},copy{};
    cudaEvent_t ready{},arrival{};
    std::vector<void*> owned;
    float *R{},*mixed{},*attn_input{},*ffn_input{},*mixer_output{},*lo{},*rs{},*xn{},*inj[2]{},*bo{},*local_out{};
    float *state{},*conv{},*candidate_state{},*candidate_conv{},*qkv{},*h{},*gate{},*beta{},*z{},*y{},*full_y{};
    float *logits{},*weights{},*sg{},*su{},*scalar{},*shared{},*parts{};
    uint8_t *xq{},*shared_local_q{},*shared_full_q{},*gu_scratch{},*down_scratch{};
    int32_t *ids{},*res{},*plan{},*keep{};
    unsigned long long* offsets{};
    uint32_t* plan_error{};
    int cap_entries, ptr_offset;
    explicit Rank(const TpGdnRankWeights& weights_in,int cap):w(weights_in),capacity(cap),
        hk(w.rank<0?16:8),hv(w.rank<0?48:24),channels(w.rank<0?C:C/2),
        value(w.rank<0?V:V/2),ff(w.rank<0?F:F/2),width(w.rank<0?N:N/2),cap_entries(cap*K) {
        select(w.device);
        try {
            check(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking),"compute stream");
            check(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking),"copy stream");
            check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming),"ready event");
            check(cudaEventCreateWithFlags(&arrival,cudaEventDisableTiming),"arrival event");
            R=alloc<float>(cap*HC*N); mixed=alloc<float>(cap*N); attn_input=alloc<float>(cap*N); ffn_input=alloc<float>(cap*N); mixer_output=alloc<float>(cap*N);
            lo=alloc<float>(cap*320);rs=alloc<float>(cap*HC);xn=alloc<float>(cap*HC*N);
            for(auto& p:inj)p=alloc<float>(cap*HC);
            bo=alloc<float>(cap*N);local_out=alloc<float>(cap*width);
            state=alloc<float>(128*hv*128);conv=alloc<float>(3*channels);
            candidate_state=alloc<float>(128*hv*128);candidate_conv=alloc<float>(3*channels);
            qkv=alloc<float>(cap*channels);h=alloc<float>(cap*channels);gate=alloc<float>(cap*hv);beta=alloc<float>(cap*hv);
            z=alloc<float>(cap*value);y=alloc<float>(cap*value);full_y=alloc<float>(cap*V);
            logits=alloc<float>(cap*NE);weights=alloc<float>(cap*K);ids=alloc<int32_t>(cap*K);
            sg=alloc<float>(cap*ff);su=alloc<float>(cap*ff);scalar=alloc<float>(cap);shared=alloc<float>(cap*width);
            parts=alloc<float>(cap*K*width);xq=alloc<uint8_t>(cap*(V/32)*36);
            shared_local_q=alloc<uint8_t>(cap*(ff/32)*36);shared_full_q=alloc<uint8_t>(cap*(F/32)*36);
            gu_scratch=alloc<uint8_t>(k::native_expert_scratch_bytes(cap_entries,ff));
            down_scratch=alloc<uint8_t>(k::native_expert_scratch_bytes(cap_entries,F));
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
    void hc(int half,int t,bool apply) {
        k::FusedGrArgs args[8]{};
        for(int j=0;j<t;++j){auto& a=args[j];a.R=R+j*HC*N;a.R_out=R+j*HC*N;a.apply=apply;
            a.bo_prev=bo+j*N;a.inj_prev=inj[0]+j*HC;a.inject_out=inj[half]+j*HC;
            a.w_norm=w.hc[half].norm;a.w_down=w.hc[half].down;a.w_up=w.hc[half].up;a.w_inject=w.hc[half].inject;
            a.lo=lo+j*320;a.rs=rs+j*HC;a.mixed=mixed+j*N;a.eps=eps;}
        k::fused_gr_read_multi(args,t,xn,compute);
    }
    void mixer(int t) {
        select(w.device);hc(0,t,false);
        check(cudaMemcpyAsync(attn_input,mixed,t*N*sizeof(float),cudaMemcpyDeviceToDevice,compute),"attention seam");
        k::native_quantize_q8_1(mixed,xq,N,t,compute);matrix(w.qkv,xq,qkv,t);matrix(w.z,xq,z,t);
        k::gdn_conv_l2_multi(conv,qkv,w.conv,h,channels,2*hk,eps,t,compute);
        k::gdn_ab_multi(mixed,w.alpha,w.beta,w.dt,w.a,gate,beta,N,hv,t,compute);
        k::gdn_step_norm_multi(state,h,channels,gate,beta,z,w.norm,eps,y,hk,hv,t,nullptr,compute);
    }
    void output_projection(int t) {
        select(w.device);k::native_quantize_q8_1(full_y,xq,V,t,compute);matrix(w.out,xq,local_out,t);
    }
    void ffn_gu(int t,int mode) {
        select(w.device);hc(1,t,true);
        check(cudaMemcpyAsync(ffn_input,mixed,t*N*sizeof(float),cudaMemcpyDeviceToDevice,compute),"FFN seam");
        k::bf16_gemv_fp32_mmvf_multi(mixed,N,w.router,logits,NE,N,NE,t,compute);
        k::native_router_top10_multi(logits,ids,weights,t,compute);
        check(cudaMemsetAsync(plan_error,0,sizeof(uint32_t),compute),"plan error reset");
        k::resident_plan(ids,t*K,K,res,NE,w.expert_arena,offsets,static_cast<long long>(w.expert_bytes),plan,cap_entries,nullptr,0,compute,plan_error);
        k::native_quantize_q8_1(mixed,xq,N,t,compute);
        matrix(w.shared_gate,xq,sg,t);matrix(w.shared_up,xq,su,t);
        k::native_swiglu_quantize_q8_1(sg,su,shared_local_q,ff,t,compute);
        k::bf16_gemv_fp32_mmvf_multi(mixed,N,w.shared_scalar,scalar,1,N,1,t,compute);
        grouped(t,mode,k::NativeExpertPhase::GateUp,w.gu_layout,gu_scratch);
    }
    void grouped(int t,int mode,k::NativeExpertPhase phase,const k::NativeExpertLayout& layout,void* scratch) {
        const int32_t* starts=plan+4;const int32_t* dst=starts+cap_entries+1;const int32_t* tok=dst+cap_entries;
        const auto* ptr=reinterpret_cast<const unsigned long long*>(plan+ptr_offset);
        k::native_expert_grouped_explicit(layout,ptr,starts,plan,dst,tok,t*K,t*K,xq,scratch,parts,compute,{phase,mode});
    }
    uint8_t* hidden(bool full,int t) const {
        return (full?down_scratch:gu_scratch)+k::native_expert_hidden_q8_offset(t*K,full?F:ff);
    }
    void ffn_down(int t,int mode) {
        select(w.device);matrix(w.shared_down,shared_full_q,shared,t);
        grouped(t,mode,k::NativeExpertPhase::Down,w.down_layout,down_scratch);
        k::native_moe_combine_multi_hits_gated(parts,weights,shared,scalar,local_out,width,K,t,compute);
    }
    void finish(int t) {
        select(w.device);k::gr_write_multi(R,bo,inj[1],{N,HC,320},R,t,compute);
    }
};
}

struct TpGdnLayer::Impl {
    ModelGeometry geometry;
    std::array<std::unique_ptr<Rank>,2> r;
    int count,capacity,mode,tokens=0;
    uint64_t epoch=0;
    bool seen_epoch=false,pending=false,poisoned=false;
    Impl(const ModelGeometry& g,const TpGdnRankWeights& a,const TpGdnRankWeights* b,int cap,int m):
        geometry(g),count(b?2:1),capacity(cap),mode(m) {
        (void)gdn_rank_layout(g,0);
        if(cap<1||cap>8)throw std::invalid_argument("TP GDN capacity must be 1..8");
        if(!k::native_expert_call_options_valid({k::NativeExpertPhase::GateUp,m}))
            throw std::invalid_argument("TP GDN invalid native expert mode");
        if((b&&(a.rank!=0||b->rank!=1||a.device==b->device||a.layer!=b->layer))||(!b&&a.rank!=-1))
            throw std::invalid_argument("TP GDN needs ranks 0,1 on different devices or rank -1 reference");
        if(a.layer<0||a.layer>=g.n_layers||is_qsa_layer(g,a.layer)||a.layer==1)
            throw std::invalid_argument("TP GDN requires an ordinary non-PLE GDN layer");
        if(!k::gr_native_mmvf_enabled())throw std::invalid_argument("TP GDN requires native GR write arithmetic configured before construction");
        if(!k::shared_expert_native_bf16_enabled())throw std::invalid_argument("TP GDN requires native FP32 shared scalar-gate arithmetic");
        const char* resident_bias=std::getenv("STRATA_ROUTE_RESIDENT");
        if(resident_bias&&std::strtof(resident_bias,nullptr)!=0.0f)throw std::invalid_argument("TP GDN does not support residency-biased routing");
        auto validate=[](const TpGdnRankWeights& w) {
            const bool full=w.rank<0;const int c=full?C:C/2,v=full?V:V/2,f=full?F:F/2,n=full?N:N/2;
            auto matrix=[](const TpNativeMatrix& x,int in,int out){if(!x.data||!k::native_mmvq_supported(x.type)||x.input!=in||x.output!=out)throw std::invalid_argument("TP GDN invalid native matrix descriptor");
                (void)k::native_mmvq_weight_bytes(x.type,in,out);};
            matrix(w.qkv,N,c);matrix(w.z,N,v);matrix(w.out,V,n);matrix(w.shared_gate,N,f);matrix(w.shared_up,N,f);matrix(w.shared_down,F,n);
            for(const auto& h:w.hc)if(!h.norm||!h.down||!h.up||!h.inject)throw std::invalid_argument("TP GDN missing HC weights");
            if(!w.alpha||!w.beta||!w.router||!w.shared_scalar||!w.dt||!w.a||!w.conv||!w.norm||!w.expert_arena||!w.expert_bytes)
                throw std::invalid_argument("TP GDN missing layer weights");
            if(w.gu_layout.n_embd!=N||w.gu_layout.n_ff!=f||w.down_layout.n_embd!=n||w.down_layout.n_ff!=F)
                throw std::invalid_argument("TP GDN expert shape mismatch");
            const auto& gu=w.gu_layout;const auto& down=w.down_layout;
            if(!k::native_expert_supported(gu.gu_type,down.d_type,N,F)||gu.bytes!=w.expert_bytes||down.bytes!=w.expert_bytes||
               gu.gu_row!=k::iq_row_bytes(gu.gu_type,N)||down.d_row!=k::iq_row_bytes(down.d_type,F)||
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
        exchange([&](Rank& d,Rank& s){
            if(count==1){bytes(d,d.full_y,s,s.y,tokens*V*sizeof(float));return;}
            // Mapped V heads are three runs of eight 128-element heads.
            for(int t=0;t<tokens;++t)for(int b=0;b<3;++b)
                bytes(d,d.full_y+t*V+(b*16+s.w.rank*8)*128,s,s.y+t*(V/2)+b*8*128,8*128*sizeof(float));
        });
    }
    void gather_output() {
        exchange([&](Rank& d,Rank& s){for(int t=0;t<tokens;++t)
            bytes(d,d.bo+t*N+(count==1?0:s.w.rank*N/2),s,s.local_out+t*s.width,s.width*sizeof(float));});
    }
    void gather_hidden() {
        exchange([&](Rank& d,Rank& s){const size_t local=(s.ff/32)*36,full=(F/32)*36,off=count==1?0:s.w.rank*local;
            for(int e=0;e<tokens*K;++e)bytes(d,d.hidden(true,tokens)+e*full+off,s,s.hidden(false,tokens)+e*local,local);
            for(int t=0;t<tokens;++t)bytes(d,d.shared_full_q+t*full+off,s,s.shared_local_q+t*local,local);
        });
    }
    void run() {
        for(int i=0;i<count;++i)r[i]->mixer(tokens);
        gather_y();for(int i=0;i<count;++i)r[i]->output_projection(tokens);
        gather_output();for(int i=0;i<count;++i){auto& a=*r[i];select(a.w.device);check(cudaMemcpyAsync(a.mixer_output,a.bo,tokens*N*sizeof(float),cudaMemcpyDeviceToDevice,a.compute),"mixer seam");a.ffn_gu(tokens,mode);}
        gather_hidden();for(int i=0;i<count;++i)r[i]->ffn_down(tokens,mode);
        gather_output();for(int i=0;i<count;++i)r[i]->finish(tokens);
        sync();
        for(int i=0;i<count;++i){uint32_t error=0;select(r[i]->w.device);check(cudaMemcpy(&error,r[i]->plan_error,sizeof(error),cudaMemcpyDeviceToHost),"plan status");
            if(error)throw std::runtime_error("TP GDN static expert plan rejected a route");}
        pending=true;
    }
    void begin(int t,uint64_t e) {
        healthy();if(pending)throw std::logic_error("TP GDN proposal requires commit(keep), including rollback with keep=0");
        if(t<1||t>capacity)throw std::invalid_argument("TP GDN token count exceeds capacity");
        if(seen_epoch&&e<=epoch)throw std::invalid_argument("TP GDN epoch must increase");
        tokens=t;epoch=e;seen_epoch=true;
    }
};

TpGdnLayer::TpGdnLayer(const ModelGeometry& g,const TpGdnRankWeights& a,const TpGdnRankWeights* b,int cap,int mode):impl_(std::make_unique<Impl>(g,a,b,cap,mode)){}
TpGdnLayer::~TpGdnLayer()=default;
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
    catch(...){p.poisoned=true;throw;}
}
void TpGdnLayer::propose_device(const std::array<const float*,2>& residual,int tokens,uint64_t epoch) {
    auto& p=*impl_;for(int i=0;i<p.count;++i)if(!residual[i])throw std::invalid_argument("TP GDN missing rank residual");
    p.begin(tokens,epoch);
    try{for(int i=0;i<p.count;++i){auto& r=*p.r[i];select(r.w.device);check(cudaMemcpyAsync(r.R,residual[i],tokens*HC*N*sizeof(float),cudaMemcpyDeviceToDevice,r.compute),"residual device copy");}p.run();}
    catch(...){p.poisoned=true;throw;}
}
void TpGdnLayer::commit(int keep) {
    auto& p=*impl_;p.healthy();if(!p.pending||keep<0||keep>p.tokens)throw std::invalid_argument("TP GDN commit requires an outstanding window and keep in 0..T");
    if(keep==0){p.pending=false;return;}
    try{for(int i=0;i<p.count;++i){auto& r=*p.r[i];select(r.w.device);
        check(cudaMemcpyAsync(r.candidate_state,r.state,128*r.hv*128*sizeof(float),cudaMemcpyDeviceToDevice,r.compute),"stage state");
        check(cudaMemcpyAsync(r.candidate_conv,r.conv,3*r.channels*sizeof(float),cudaMemcpyDeviceToDevice,r.compute),"stage conv");
        check(cudaMemcpyAsync(r.keep,&keep,sizeof(keep),cudaMemcpyHostToDevice,r.compute),"commit count");
        k::gdn_conv_commit(r.candidate_conv,r.qkv,r.channels,r.keep,r.compute);
        k::gdn_step_norm_multi(r.candidate_state,r.h,r.channels,r.gate,r.beta,r.z,r.w.norm,eps,r.y,r.hk,r.hv,p.tokens,r.keep,r.compute,p.tokens);
    }p.sync();
    // Pointer publication is all-host, after both successful device completions.
    for(int i=0;i<p.count;++i){std::swap(p.r[i]->state,p.r[i]->candidate_state);std::swap(p.r[i]->conv,p.r[i]->candidate_conv);}p.pending=false;
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
    get(s.gdn_output,r.full_y,p.tokens*V);get(s.output,r.bo,p.tokens*N);get(s.ids,r.ids,p.tokens*K);get(s.route_weights,r.weights,p.tokens*K);
    get(s.mixer_output,r.mixer_output,p.tokens*N);get(s.route_logits,r.logits,p.tokens*NE);
    get(s.shared_gate_logits,r.scalar,p.tokens);
    s.shared_output.resize(p.tokens*N);s.expert_parts.resize(p.tokens*K*N);
    for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);std::vector<float> sh,part;get(sh,a.shared,p.tokens*a.width);get(part,a.parts,p.tokens*K*a.width);
        const int offset=p.count==1?0:i*a.width;
        for(int t=0;t<p.tokens;++t)std::copy_n(sh.data()+t*a.width,a.width,s.shared_output.data()+t*N+offset);
        for(int e=0;e<p.tokens*K;++e)std::copy_n(part.data()+e*a.width,a.width,s.expert_parts.data()+e*N+offset);
    }
    select(r.w.device);
    get(s.routed_hidden_q8,r.hidden(true,p.tokens),static_cast<size_t>(p.tokens)*K*(F/32)*36);
    get(s.shared_hidden_q8,r.shared_full_q,static_cast<size_t>(p.tokens)*(F/32)*36);
    s.state.resize(128u*48u*128u);s.conv.resize(3u*C);
    for(int i=0;i<p.count;++i){auto& a=*p.r[i];select(a.w.device);std::vector<float> st,cv;get(st,a.state,128*a.hv*128);get(cv,a.conv,3*a.channels);
        if(p.count==1){s.state=std::move(st);s.conv=std::move(cv);}else{const auto map=gdn_rank_layout(p.geometry,i);
            for(int row=0;row<128;++row)for(int h=0;h<24;++h)for(int col=0;col<128;++col)s.state[map.global_state_index(row,h,col)]=st[map.local_state_index(row,h,col)];
            for(int tap=0;tap<3;++tap)for(int c=0;c<a.channels;++c)s.conv[map.qkv_rows[c]*3+tap]=cv[c*3+tap];}
    }
    return s;
}
} // namespace strata::core::tp2
