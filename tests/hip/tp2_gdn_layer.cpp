// Opt-in real-weight whole-GDN-layer diagnostic. Requires two peer-accessible GPUs.
// The -1 reference shares executor orchestration, NOT an independent Verifier.
// A separate token-serial legacy fused-GDN replay checks its recurrence/commit.
// Fixed numerical gates below are engineering parity gates, not model quality.
#include "strata/core/tp_gdn_layer.hpp"
#include "strata/core/tp_gdn_weights.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tp = strata::core::tp2;
namespace k = strata::kernels;
namespace {
constexpr int N=2560, H=4, S=128, HK=16, HV=48, C=10240, V=6144, F=640, K=10;
constexpr size_t STATE=size_t(S)*HV*S, CONV=size_t(C)*3;
void ck(cudaError_t e,const char* where) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
void on(int d) { ck(cudaSetDevice(d),"set device"); }
struct Options {
    std::vector<std::string> shards; std::string pack,execution="runtime";
    int layer=0,mode=8,warmup=3,trials=12; bool benchmark=false;
};
Options parse(int argc,char**argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--benchmark"){o.benchmark=true;continue;}
        if(i+1==argc) throw std::invalid_argument("missing argument for "+a);
        const std::string v=argv[++i];
        if(a=="--gguf") o.shards.push_back(v);
        else if(a=="--pack") o.pack=v;
        else if(a=="--execution")o.execution=v;
        else if(a=="--layer" || a=="--mode" || a=="--bench-warmup" || a=="--bench-trials") {
            size_t used=0; int n=std::stoi(v,&used);
            if(used!=v.size()) throw std::invalid_argument("invalid integer");
            if(a=="--layer")o.layer=n;else if(a=="--mode")o.mode=n;else if(a=="--bench-warmup")o.warmup=n;else o.trials=n;
        } else throw std::invalid_argument("unknown option "+a);
    }
    if(o.shards.empty() || o.pack.empty() || o.layer<0 || o.layer>=48 || o.layer==1 || o.layer%4==3 ||
       (o.mode!=7 && o.mode!=8) || o.warmup<1 || o.warmup>100 || o.trials<2 || o.trials>1000 ||
       (o.execution!="runtime"&&o.execution!="consolidated"&&o.execution!="captured"&&o.execution!="all"))
        throw std::invalid_argument("usage: tp2_gdn_layer --pack PACK --gguf SHARD [--gguf SHARD ...] [--layer non-PLE-GDN-index] [--mode 7|8] [--execution runtime|consolidated|captured|all] [--benchmark] [--bench-warmup 3] [--bench-trials 12]");
    return o;
}
template<class T> std::vector<T> read(const T* p,size_t n,int device) {
    on(device); std::vector<T> v(n);
    if(n) ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"diagnostic download");
    return v;
}
template<class T> struct Buffer {
    T* p=nullptr; size_t n; int device;
    Buffer(size_t count,int d):n(count),device(d) {on(d);ck(cudaMalloc((void**)&p,n*sizeof(T)),"legacy buffer");}
    ~Buffer(){if(p){cudaSetDevice(device);cudaFree(p);}}
    Buffer(const Buffer&)=delete; Buffer&operator=(const Buffer&)=delete;
    void set(const std::vector<T>& v) {if(v.size()!=n)throw std::runtime_error("buffer shape");on(device);ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice),"legacy upload");}
    std::vector<T> get() const {return read(p,n,device);}
};
struct Stream {cudaStream_t s{};int device;explicit Stream(int d):device(d){on(d);ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"legacy stream");}~Stream(){cudaSetDevice(device);cudaStreamDestroy(s);}};
struct Gate {double abs,scaled,rms;};
// max error <= abs + scaled * reference RMS; RMS error <= abs + rms * reference RMS.
// These tolerances are frozen before hardware measurement, never fitted to output.
constexpr Gate hc_gate{2e-6,2e-5,5e-6}, gdn_gate{3e-6,5e-5,1e-5}, ffn_gate{1e-5,3e-4,5e-5};
struct Checks {
    int failures=0;
    void floats(const char* label,const std::vector<float>& a,const std::vector<float>& b,Gate g) {
        if(a.size()!=b.size()||a.empty()){++failures;std::printf("FAIL %-27s shape %zu/%zu\n",label,a.size(),b.size());return;}
        double sq=0,err=0,max=0;bool finite=true;
        for(size_t i=0;i<a.size();++i){finite &= std::isfinite(a[i])&&std::isfinite(b[i]);const double d=double(a[i])-b[i];sq+=double(a[i])*a[i];err+=d*d;max=std::max(max,std::abs(d));}
        const double ref=std::sqrt(sq/a.size()),rms=std::sqrt(err/a.size());
        const bool pass=finite&&max<=g.abs+g.scaled*ref&&rms<=g.abs+g.rms*ref;
        if(!pass)++failures;
        std::printf("%s %-27s n=%zu ref_rms=%.7g max=%.7g rms=%.7g limits=%.7g/%.7g\n",pass?"PASS":"FAIL",label,a.size(),ref,max,rms,g.abs+g.scaled*ref,g.abs+g.rms*ref);
    }
    template<class T> void exact(const char* label,const std::vector<T>& a,const std::vector<T>& b) {
        size_t different=0;
        if(a.size()==b.size()) for(size_t i=0;i<a.size();++i)different+=std::memcmp(&a[i],&b[i],sizeof(T))!=0;
        const bool pass=!a.empty()&&a.size()==b.size()&&!different;
        if(!pass)++failures;
        std::printf("%s %-27s n=%zu/%zu different=%zu\n",pass?"PASS":"FAIL",label,a.size(),b.size(),different);
    }
    void require(bool pass,const char* label){if(!pass)++failures;std::printf("%s %s\n",pass?"PASS":"FAIL",label);}
};
std::vector<float> signed_input(size_t n,int salt,float scale) {
    // Nonzero deterministic signed values, head/channel/tap-varying. Integer
    // construction avoids implementation-defined random distributions.
    std::vector<float> v(n);
    for(size_t i=0;i<n;++i){const int z=int((i*73+size_t(salt)*131+(i/128)*29)%2003)-1001;v[i]=scale*float(z==0?1:z)/1001.f;}
    return v;
}
void seams(Checks& c,const tp::TpGdnLayerSnapshot& a,const tp::TpGdnLayerSnapshot& b,bool replica=false) {
    c.require(a.tokens==b.tokens&&a.epoch==b.epoch,"snapshot token/epoch identity");
    if(replica)c.exact("replica final residual",a.residual,b.residual);
    else c.floats("final residual",a.residual,b.residual,ffn_gate);
    if(replica) {
        c.exact("replica HC attention input",a.attention_input,b.attention_input);
        c.exact("replica GDN output",a.gdn_output,b.gdn_output);
        c.exact("replica mixer",a.mixer_output,b.mixer_output);
        c.exact("replica HC FFN input",a.ffn_input,b.ffn_input);
        c.exact("replica combined FFN",a.output,b.output);
    } else {
        c.floats("HC attention input",a.attention_input,b.attention_input,hc_gate);
        c.floats("GDN output",a.gdn_output,b.gdn_output,gdn_gate);
        c.floats("mixer projection",a.mixer_output,b.mixer_output,gdn_gate);
        c.floats("HC FFN input",a.ffn_input,b.ffn_input,hc_gate);
        c.floats("combined FFN",a.output,b.output,ffn_gate);
    }
    c.exact("router logits",a.route_logits,b.route_logits);
    c.exact("ordered router IDs",a.ids,b.ids);
    c.exact("router weights",a.route_weights,b.route_weights);
    c.exact("routed hidden Q8 bytes",a.routed_hidden_q8,b.routed_hidden_q8);
    c.exact("shared hidden Q8 bytes",a.shared_hidden_q8,b.shared_hidden_q8);
    c.floats("committed recurrence",a.state,b.state,replica?Gate{0,0,0}:gdn_gate);
    c.floats("committed conv",a.conv,b.conv,replica?Gate{0,0,0}:gdn_gate);
}
std::vector<int> sequence(int first,int count) {std::vector<int> r(count);std::iota(r.begin(),r.end(),first);return r;}
void mapped_bytes(Checks& c,const char* label,const void* full,const void* shard,size_t rowbytes,
                  const std::vector<int>& rows,int full_device,int shard_device) {
    const auto a=read(static_cast<const uint8_t*>(full),(size_t(*std::max_element(rows.begin(),rows.end()))+1)*rowbytes,full_device);
    const auto b=read(static_cast<const uint8_t*>(shard),rows.size()*rowbytes,shard_device);
    std::vector<uint8_t> expected(b.size());
    for(size_t i=0;i<rows.size();++i)std::memcpy(expected.data()+i*rowbytes,a.data()+size_t(rows[i])*rowbytes,rowbytes);
    c.exact(label,expected,b);
}
void coherence(Checks& c,const strata::core::ModelGeometry& g,const tp::TpGdnRankWeights& full,const tp::TpGdnRankWeights& r) {
    // Expected mapping is written independently of gdn_rank_layout(). A
    // self-consistent but contiguous V split must fail this actual-byte gate.
    std::vector<int> heads,qkv,value;
    for(int h=0;h<24;++h)heads.push_back(16*(h/8)+8*r.rank+h%8);
    for(int group=0;group<2;++group)for(int h=0;h<8;++h)for(int j=0;j<128;++j)qkv.push_back(group*2048+(r.rank*8+h)*128+j);
    for(int h:heads)for(int j=0;j<128;++j){qkv.push_back(4096+h*128+j);value.push_back(h*128+j);}
    auto matrix=[&](const char* label,const tp::TpNativeMatrix& a,const tp::TpNativeMatrix& b,const std::vector<int>& rows){
        c.require(a.type==b.type&&a.input==b.input&&b.output==int(rows.size()),label);
        mapped_bytes(c,label,a.data,b.data,k::native_mmvq_weight_bytes(a.type,a.input,1),rows,full.device,r.device);
    };
    matrix("actual QKV mapped rows",full.qkv,r.qkv,qkv);
    matrix("actual Z mapped rows",full.z,r.z,value);
    matrix("actual output rows",full.out,r.out,sequence(r.rank*N/2,N/2));
    matrix("actual shared gate rows",full.shared_gate,r.shared_gate,sequence(r.rank*F/2,F/2));
    matrix("actual shared up rows",full.shared_up,r.shared_up,sequence(r.rank*F/2,F/2));
    matrix("actual shared down rows",full.shared_down,r.shared_down,sequence(r.rank*N/2,N/2));
    mapped_bytes(c,"actual alpha head rows",full.alpha,r.alpha,N*2,heads,full.device,r.device);
    mapped_bytes(c,"actual beta head rows",full.beta,r.beta,N*2,heads,full.device,r.device);
    mapped_bytes(c,"actual dt head values",full.dt,r.dt,4,heads,full.device,r.device);
    mapped_bytes(c,"actual A head values",full.a,r.a,4,heads,full.device,r.device);
    mapped_bytes(c,"actual convolution rows",full.conv,r.conv,4*4,qkv,full.device,r.device);
    c.exact("actual replicated norm",read(full.norm,S,full.device),read(r.norm,S,r.device));
    c.require(g.ssm_k_heads==HK&&g.ssm_v_heads==HV,"expected actual head geometry");
}
struct LegacyGdnResult {std::vector<float> output,mixer,state,conv;};
LegacyGdnResult legacy_gdn(const tp::TpGdnRankWeights& w,const std::vector<float>& input,
                          const std::vector<float>& state,const std::vector<float>& conv,int tokens,int keep) {
    // Independent one-token loop follows fused_gdn.hpp and the original
    // Verifier GDN sequence, rather than TpGdnLayer's multi-token orchestrator.
    Stream st(w.device);Buffer<float>x(input.size(),w.device),s(STATE,w.device),hist(CONV,w.device);
    Buffer<float>qkv(C,w.device),h(C,w.device),a(HV,w.device),b(HV,w.device),z(V,w.device),y(V,w.device),out(N,w.device);
    Buffer<uint8_t>xq(k::native_q8_1_bytes(V),w.device);
    x.set(input);s.set(state);hist.set(conv);
    LegacyGdnResult result;result.state=state;result.conv=conv;
    for(int t=0;t<tokens;++t) {
        on(w.device);const float* xt=x.p+size_t(t)*N;
        k::native_quantize_q8_1(xt,xq.p,N,1,st.s);
        k::native_mmvq(w.qkv.type,w.qkv.data,xq.p,qkv.p,N,C,1,st.s);
        k::fused_gdn_conv_l2(hist.p,qkv.p,w.conv,h.p,C,2*HK,1e-6f,st.s);
        k::fused_gdn_ab(xt,w.alpha,w.beta,w.dt,w.a,a.p,b.p,N,HV,st.s);
        k::native_mmvq(w.z.type,w.z.data,xq.p,z.p,N,V,1,st.s);
        k::fused_gdn_step_norm(s.p,h.p,h.p+S*HK,h.p+2*S*HK,a.p,b.p,z.p,w.norm,1e-6f,y.p,HK,HV,st.s);
        k::native_quantize_q8_1(y.p,xq.p,V,1,st.s);
        k::native_mmvq(w.out.type,w.out.data,xq.p,out.p,V,N,1,st.s);
        ck(cudaStreamSynchronize(st.s),"legacy GDN token");
        const auto yy=y.get(),oo=out.get();result.output.insert(result.output.end(),yy.begin(),yy.end());result.mixer.insert(result.mixer.end(),oo.begin(),oo.end());
        if(t+1==keep){result.state=s.get();result.conv=hist.get();}
    }
    return result;
}
float bf16(uint16_t b) {uint32_t bits=uint32_t(b)<<16;float f;std::memcpy(&f,&bits,4);return f;}
void route_margins(Checks& c,const tp::TpGdnRankWeights& w,const tp::TpGdnLayerSnapshot& s) {
    const auto router=read(w.router,size_t(512)*N,w.device);
    for(int t=0;t<s.tokens;++t) {
        std::array<double,512> logits{};std::array<int,512> order{};std::iota(order.begin(),order.end(),0);
        for(int e=0;e<512;++e)for(int j=0;j<N;++j)logits[e]+=double(bf16(router[size_t(e)*N+j]))*s.ffn_input[size_t(t)*N+j];
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return logits[a]>logits[b];});
        const double boundary=logits[order[9]]-logits[order[10]];
        std::array<int,512> gpu_order{};std::iota(gpu_order.begin(),gpu_order.end(),0);
        bool finite_logits=true;for(int e=0;e<512;++e)finite_logits&=std::isfinite(s.route_logits[size_t(t)*512+e]);
        c.require(finite_logits,"router logits finite");
        if(finite_logits){std::stable_sort(gpu_order.begin(),gpu_order.end(),[&](int a,int b){return s.route_logits[size_t(t)*512+a]>s.route_logits[size_t(t)*512+b];});
            const double gap=double(s.route_logits[size_t(t)*512+gpu_order[9]])-s.route_logits[size_t(t)*512+gpu_order[10]];
            std::printf("router-margin token=%d actual-device-F32 top10/11=%.9g\n",t,gap);}
        double adjacent=std::numeric_limits<double>::infinity();for(int i=0;i<9;++i)adjacent=std::min(adjacent,logits[order[i]]-logits[order[i+1]]);
        int mismatches=0;for(int i=0;i<K;++i)mismatches+=s.ids[size_t(t)*K+i]!=order[i];
        double sum=0;bool valid=true;for(int i=0;i<K;++i){const float v=s.route_weights[size_t(t)*K+i];valid&=std::isfinite(v)&&v>=0;sum+=v;}
        c.require(valid&&std::abs(sum-1)<2e-6,"router weights finite nonnegative normalized");
        // CPU F64 reduction differs from device F32 and softmax tie rounding;
        // diagnostic only, while TP vs full ordered routing is a strict gate.
        std::printf("router-margin token=%d CPU-F64 top10/11=%.9g min-top10-adjacent=%.9g CPU/GPU-ordered-differences=%d (diagnostic, not a quality gate)\n",t,boundary,adjacent,mismatches);
    }
}
void legacy_hc(Checks& c,const tp::TpGdnRankWeights& w,const std::vector<float>& input,const tp::TpGdnLayerSnapshot& s) {
    Stream st(w.device);const k::GrShapes shape{N,H,320};
    Buffer<uint8_t>scratch(k::gr_workspace_bytes(shape),w.device);k::GrWorkspace ws;k::gr_workspace_init(shape,scratch.p,ws);
    Buffer<float>r(size_t(s.tokens)*H*N,w.device),mixed(size_t(s.tokens)*N,w.device),inject(size_t(s.tokens)*H,w.device),bo(size_t(s.tokens)*N,w.device);
    r.set(input);std::vector<float> inj;
    for(int half=0;half<2;++half){
        const auto& h=w.hc[half];
        for(int t=0;t<s.tokens;++t)k::gr_read(r.p+size_t(t)*H*N,h.norm,h.down,h.up,h.inject,1e-6f,shape,ws,mixed.p+size_t(t)*N,inject.p+size_t(t)*H,st.s);
        ck(cudaStreamSynchronize(st.s),"legacy HC read");
        c.floats(half?"legacy HC FFN read":"legacy HC attention read",half?s.ffn_input:s.attention_input,mixed.get(),hc_gate);
        bo.set(half?s.output:s.mixer_output);inj=inject.get();
        const auto before=r.get();
        for(int t=0;t<s.tokens;++t)k::gr_write(r.p+size_t(t)*H*N,bo.p+size_t(t)*N,inject.p+size_t(t)*H,shape,r.p+size_t(t)*H*N,st.s);
        ck(cudaStreamSynchronize(st.s),"legacy HC write");
        const auto after=r.get();std::vector<float> formula(after.size());const auto& block=half?s.output:s.mixer_output;
        for(int t=0;t<s.tokens;++t)for(int hidx=0;hidx<H;++hidx)for(int j=0;j<N;++j){
            const size_t pos=(size_t(t)*H+hidx)*N+j;const double gate=2.0/(1.0+std::exp(-double(inj[size_t(t)*H+hidx])/H));
            formula[pos]=float(double(before[pos])+gate*block[size_t(t)*N+j]);
        }
        c.floats("HC write independent formula",formula,after,hc_gate);
    }
    c.floats("legacy final HC residual",s.residual,r.get(),ffn_gate);
}
void legacy_shared_and_combine(Checks& c,const tp::TpGdnRankWeights& w,const tp::TpGdnLayerSnapshot& s) {
    Stream st(w.device);Buffer<float>x(s.ffn_input.size(),w.device),gate(size_t(s.tokens)*F,w.device),up(size_t(s.tokens)*F,w.device),scalar(s.tokens,w.device),out(size_t(s.tokens)*N,w.device);
    Buffer<uint8_t>q(k::native_q8_1_bytes(N,s.tokens),w.device);x.set(s.ffn_input);
    k::NativeSharedWeights nw;nw.gate_type=w.shared_gate.type;nw.up_type=w.shared_up.type;nw.down_type=w.shared_down.type;
    nw.gate_data=w.shared_gate.data;nw.up_data=w.shared_up.data;nw.down_data=w.shared_down.data;nw.q8_1=q.p;
    k::shared_expert_multi(s.tokens,x.p,nullptr,nw,w.shared_scalar,gate.p,up.p,scalar.p,out.p,N,F,st.s,nullptr,0);
    ck(cudaStreamSynchronize(st.s),"legacy shared expert");
    auto shared=out.get();std::vector<float> expected(shared.size()),combined(shared.size());
    for(int t=0;t<s.tokens;++t)for(int j=0;j<N;++j){
        const size_t p=size_t(t)*N+j;const double scalar_gate=1.0/(1.0+std::exp(-double(s.shared_gate_logits[t])));
        expected[p]=float(scalar_gate*s.shared_output[p]);
        double sum=0;for(int e=0;e<K;++e)sum+=double(s.route_weights[size_t(t)*K+e])*s.expert_parts[(size_t(t)*K+e)*N+j];
        combined[p]=float(sum+expected[p]);
    }
    c.floats("legacy shared scaled output",expected,shared,ffn_gate);
    // After the legacy shared call the same scratch starts with its full-F Q8.
    c.exact("legacy shared hidden Q8",s.shared_hidden_q8,read(q.p,size_t(s.tokens)*(F/32)*36,w.device));
    c.floats("CPU FFN combine formula",combined,s.output,ffn_gate);
}
void legacy_routed(Checks& c,const tp::TpGdnRankWeights& w,const tp::TpGdnLayerSnapshot& s) {
    Stream st(w.device);Buffer<float>x(s.ffn_input.size(),w.device),out(N,w.device);
    Buffer<uint8_t>q(k::native_q8_1_bytes(N,s.tokens),w.device),scratch(k::native_expert_scratch_bytes(1,F),w.device);
    Buffer<unsigned long long>ptr(1,w.device);Buffer<int32_t>start(2,w.device),groups(1,w.device),dst(1,w.device),tok(1,w.device);
    x.set(s.ffn_input);start.set({0,1});groups.set({1});dst.set({0});
    k::native_quantize_q8_1(x.p,q.p,N,s.tokens,st.s);ck(cudaStreamSynchronize(st.s),"legacy routed quantize");
    std::vector<float> parts;std::vector<std::vector<uint8_t>> hidden(s.ids.size());
    for(size_t e=0;e<s.ids.size();++e){
        if(s.ids[e]<0||s.ids[e]>=512)throw std::runtime_error("router ID outside expert arena");
        ptr.set({reinterpret_cast<unsigned long long>(w.expert_arena+size_t(s.ids[e])*w.expert_bytes)});tok.set({int32_t(e/K)});
        // Original full-FFN API, independent host metadata, one selected entry
        // at a time. The executor uses explicit GU and Down phase calls.
        k::native_expert_grouped(w.gu_layout,ptr.p,start.p,groups.p,dst.p,tok.p,1,1,q.p,scratch.p,out.p,st.s,1);
        ck(cudaStreamSynchronize(st.s),"legacy routed expert");
        auto row=out.get();parts.insert(parts.end(),row.begin(),row.end());
        hidden[e]=read(scratch.p+k::native_expert_hidden_q8_offset(1,F),size_t(F/32)*36,w.device);
    }
    c.floats("legacy full routed parts",parts,s.expert_parts,ffn_gate);
    // resident_plan orders groups by first occurrence and entries within each
    // group by original top-k position; reconstruct that order independently.
    std::vector<uint8_t> ordered;std::array<bool,512> seen{};
    for(size_t e=0;e<s.ids.size();++e)if(!seen[s.ids[e]]){seen[s.ids[e]]=true;
        for(size_t j=e;j<s.ids.size();++j)if(s.ids[j]==s.ids[e])ordered.insert(ordered.end(),hidden[j].begin(),hidden[j].end());}
    c.exact("legacy routed hidden Q8",ordered,s.routed_hidden_q8);
}
void committed(Checks& c,const tp::TpGdnLayerSnapshot& a,const tp::TpGdnLayerSnapshot& b,const LegacyGdnResult& legacy) {
    c.floats("commit full/TP recurrence",a.state,b.state,gdn_gate);
    c.floats("commit full/TP conv",a.conv,b.conv,gdn_gate);
    c.floats("commit legacy recurrence",legacy.state,b.state,gdn_gate);
    c.floats("commit legacy conv",legacy.conv,b.conv,gdn_gate);
}

const char* execution_name(tp::TpGdnExecution mode) {
    switch(mode){case tp::TpGdnExecution::RuntimeCopies:return "runtime";case tp::TpGdnExecution::Consolidated:return "consolidated";case tp::TpGdnExecution::Captured:return "captured";}
    throw std::invalid_argument("unknown execution mode");
}
std::vector<tp::TpGdnExecution> execution_modes(const Options& o) {
    if(o.benchmark||o.execution=="all")return {tp::TpGdnExecution::RuntimeCopies,tp::TpGdnExecution::Consolidated,tp::TpGdnExecution::Captured};
    if(o.execution=="captured")return {tp::TpGdnExecution::Captured};
    if(o.execution=="consolidated")return {tp::TpGdnExecution::Consolidated};
    return {tp::TpGdnExecution::RuntimeCopies};
}
struct LayerTime {double propose_ms=0,commit_ms=0,total_ms=0;};
LayerTime timed_layer(tp::TpGdnLayer& layer,const std::array<const float*,2>& input,int tokens,uint64_t epoch) {
    using Clock=std::chrono::steady_clock;
    // Both APIs synchronize their own rank streams before return. The wall
    // interval includes D2D input copies, host orchestration, launches, peer
    // exchange, whole-layer compute, accepted-prefix replay and synchronization.
    // No device-wide sync, diagnostic download, reset/upload or graph build.
    const auto start=Clock::now();layer.propose_device(input,tokens,epoch);
    const auto proposed=Clock::now();layer.commit(tokens);const auto end=Clock::now();
    auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
    return {ms(start,proposed),ms(proposed,end),ms(start,end)};
}
double quantile(std::vector<double> values,double q) {
    std::sort(values.begin(),values.end());const double x=q*double(values.size()-1);
    const size_t i=size_t(x),j=std::min(i+1,values.size()-1);return values[i]+(values[j]-values[i])*(x-double(i));
}
void benchmark(Checks& c,const Options& o,tp::TpGdnLayer& reference,tp::TpGdnLayer& parallel,
               uint64_t& epoch,const std::vector<float>& state,const std::vector<float>& conv) {
    std::printf("\nBENCH_SCOPE real-weight-one-GDN-layer; baseline=one-GPU-unsharded; candidate=two-GPU-TP; same-execution-mode-per-pair; not-EP-or-full-model-generation\n");
    std::printf("BENCH_METHOD wall=propose_device+commit(T), includes D2D input/host launch/P2P/compute/commit/sync; excludes reset/H2D/weight-load/graph-preparation/snapshots; fixed initialized state, varying matched signed input; keep=T is a workload choice, not measured speculative acceptance\n");
    std::printf("BENCH_CONTRACT both arms use speculative proposal plus explicit commit replay, including T=1; this is not optimized autoregressive self-commit or the production Verifier/EP baseline; fixed native-GR/native-BF16-shared/exact-MMVQ settings and explicit expert mode=%d\n",o.mode);
    std::printf("BENCH_LIMIT execution modes run in separate blocks; only same-mode full/TP trials are paired, so cross-mode absolute timing differences are diagnostic and may include thermal/clock drift\n");
    std::printf("BENCH_CONFIG warmup_pairs=%d measured_pairs=%d order=alternating per-T; parity checked after every pair outside timer; summaries conditional on all gates passing\n",o.warmup,o.trials);
    // Prepare each graph once, then reuse across later T changes. Preparation is
    // never part of a timing sample and has no claimed amortization horizon.
    for(int t:{1,2,4,8}){reference.prepare_captured(t);parallel.prepare_captured(t);}
    for(auto mode:execution_modes(o))for(int tokens:{1,2,4,8}){
        reference.set_execution(mode);parallel.set_execution(mode);
        Buffer<float>input0(size_t(tokens)*H*N,0),input1(size_t(tokens)*H*N,1);
        std::vector<double>rt,pt,ratio,rp,pp,rc,pc;
        for(int trial=-o.warmup;trial<o.trials;++trial){
            const auto input=signed_input(size_t(tokens)*H*N,1201+tokens*97+(trial+o.warmup)*31,0.65f);
            input0.set(input);input1.set(input);const std::array<const float*,2> inputs{input0.p,input1.p};
            const uint64_t pair_epoch=++epoch;const bool tp_first=((trial+o.warmup)&1)!=0;
            LayerTime a,b;
            auto one=[&](tp::TpGdnLayer& layer){layer.reset_state(state,conv);return timed_layer(layer,inputs,tokens,pair_epoch);};
            if(tp_first){b=one(parallel);a=one(reference);}else{a=one(reference);b=one(parallel);}
            std::printf("BENCH_%s mode=%s T=%d trial=%d first=%s full-propose-ms=%.6f full-commit-ms=%.6f full-total-ms=%.6f tp-propose-ms=%.6f tp-commit-ms=%.6f tp-total-ms=%.6f\n",trial<0?"WARMUP":"SAMPLE",execution_name(mode),tokens,trial,tp_first?"TP":"full",a.propose_ms,a.commit_ms,a.total_ms,b.propose_ms,b.commit_ms,b.total_ms);
            // Never interleave snapshot downloads or checks between timed arms.
            const auto full=reference.snapshot(),split=parallel.snapshot();seams(c,full,split);seams(c,split,parallel.snapshot(1),true);
            if(c.failures)throw std::runtime_error("benchmark parity failed; timing is not an accepted result");
            if(trial>=0){rt.push_back(a.total_ms);pt.push_back(b.total_ms);ratio.push_back(a.total_ms/b.total_ms);rp.push_back(a.propose_ms);pp.push_back(b.propose_ms);rc.push_back(a.commit_ms);pc.push_back(b.commit_ms);}
        }
        std::printf("BENCH_SUMMARY mode=%s T=%d pairs=%zu full-ms-median=%.6f full-ms-p10=%.6f full-ms-p90=%.6f tp-ms-median=%.6f tp-ms-p10=%.6f tp-ms-p90=%.6f paired-ratio-median=%.6f ratio-of-total-times=%.6f full-propose-median=%.6f tp-propose-median=%.6f full-commit-median=%.6f tp-commit-median=%.6f\n",execution_name(mode),tokens,rt.size(),quantile(rt,.5),quantile(rt,.1),quantile(rt,.9),quantile(pt,.5),quantile(pt,.1),quantile(pt,.9),quantile(ratio,.5),std::accumulate(rt.begin(),rt.end(),0.0)/std::accumulate(pt.begin(),pt.end(),0.0),quantile(rp,.5),quantile(pp,.5),quantile(rc,.5),quantile(pc,.5));
        std::fflush(stdout);
    }
}
} // namespace
int main(int argc,char** argv) {
    try {
        const auto o=parse(argc,argv);int devices=0;ck(cudaGetDeviceCount(&devices),"device count");
        if(devices<2)throw std::runtime_error("two GPUs required; this is not a CPU or single-GPU pass");
        for(int d=0;d<2;++d){cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,d),"device properties");std::printf("device=%d name=%s\n",d,prop.name);int access=0;ck(cudaDeviceCanAccessPeer(&access,d,1-d),"P2P check");if(!access)throw std::runtime_error("bidirectional P2P required");}
        k::gr_set_native_mmvf(true);k::shared_expert_set_native_bf16(true);k::native_mmvq_set_multi_exact(true);k::native_expert_set_mode(o.mode,0);
        strata::core::ModelGeometry g;tp::TpGdnWeights full,rank0,rank1;std::string err;
        if(!full.load(o.shards,o.pack,g,o.layer,-1,0,err))throw std::runtime_error("full reference load: "+err);
        if(!rank0.load(o.shards,o.pack,g,o.layer,0,0,err))throw std::runtime_error("rank0 load: "+err);
        if(!rank1.load(o.shards,o.pack,g,o.layer,1,1,err))throw std::runtime_error("rank1 load: "+err);
        std::printf("real-weight GDN layer=%d mode=%d weight-bytes full=%llu rank0=%llu rank1=%llu\n",o.layer,o.mode,(unsigned long long)full.weight_bytes(),(unsigned long long)rank0.weight_bytes(),(unsigned long long)rank1.weight_bytes());
        std::printf("scope=one-layer; full-reference shares orchestrator; independent legacy GDN/HC/shared and CPU-combine checks; no model-quality or performance claim\n");
        Checks c;coherence(c,g,full.weights(),rank0.weights());coherence(c,g,full.weights(),rank1.weights());
        tp::TpGdnLayer reference(g,full.weights(),nullptr,8,o.mode),parallel(g,rank0.weights(),&rank1.weights(),8,o.mode);
        const auto executions=execution_modes(o);
        if(std::find(executions.begin(),executions.end(),tp::TpGdnExecution::Captured)!=executions.end()){
            // New graph shapes are initialization-only, before any proposal.
            for(int t=1;t<=8;++t)parallel.prepare_captured(t);
            if(o.benchmark)for(int t:{1,2,4,8})reference.prepare_captured(t);
        }
        uint64_t epoch=0;int cases=0;
        const auto state=signed_input(STATE,17,0.015f),conv=signed_input(CONV,31,0.12f);
        std::vector<int32_t> prior_routes;bool routes_changed=false,repeated_ids=false;
        for(auto execution:executions){
        parallel.set_execution(execution);
        std::printf("\nCORRECTNESS_EXECUTION %s (reference=runtime)\n",execution_name(execution));
        for(int tokens=1;tokens<=8;++tokens)for(int keep=0;keep<=tokens;++keep){
            std::printf("\nCASE T=%d keep=%d epoch=%llu\n",tokens,keep,(unsigned long long)(epoch+1));std::fflush(stdout);
            const auto input=signed_input(size_t(tokens)*H*N,43+tokens*19+keep*7,0.65f);
            reference.reset_state(state,conv);parallel.reset_state(state,conv);
            reference.propose(input,tokens,++epoch);parallel.propose(input,tokens,epoch);
            auto a=reference.snapshot(),b=parallel.snapshot();seams(c,a,b);seams(c,b,parallel.snapshot(1),true);
            c.exact("proposal leaves state",state,b.state);c.exact("proposal leaves conv",conv,b.conv);
            if(!prior_routes.empty()&&std::vector<int32_t>(b.ids.begin(),b.ids.begin()+K)!=prior_routes)routes_changed=true;
            prior_routes.assign(b.ids.begin(),b.ids.begin()+K);
            {std::array<bool,512> seen{};for(int id:b.ids){if(id<0||id>=512)throw std::runtime_error("invalid route ID");repeated_ids|=seen[id];seen[id]=true;}}
            route_margins(c,full.weights(),a);
            const auto legacy=legacy_gdn(full.weights(),a.attention_input,state,conv,tokens,keep);
            c.floats("serial legacy GDN output",legacy.output,b.gdn_output,gdn_gate);
            c.floats("serial legacy mixer output",legacy.mixer,b.mixer_output,gdn_gate);
            legacy_hc(c,full.weights(),input,b);legacy_shared_and_combine(c,full.weights(),b);legacy_routed(c,full.weights(),b);
            reference.commit(keep);parallel.commit(keep);a=reference.snapshot();b=parallel.snapshot();committed(c,a,b,legacy);
            // A new input after EVERY keep catches accidental full-window commit,
            // rejected-tail leakage, stale exchange slots and zero-keep rollback.
            const auto next=signed_input(size_t(2)*H*N,101+tokens*13+keep*23,0.55f);
            reference.propose(next,2,++epoch);parallel.propose(next,2,epoch);
            const auto na=reference.snapshot(),nb=parallel.snapshot();seams(c,na,nb);seams(c,nb,parallel.snapshot(1),true);
            const auto continuation=legacy_gdn(full.weights(),na.attention_input,legacy.state,legacy.conv,2,2);
            c.floats("continuation legacy GDN",continuation.output,nb.gdn_output,gdn_gate);
            c.floats("continuation legacy mixer",continuation.mixer,nb.mixer_output,gdn_gate);
            reference.commit(2);parallel.commit(2);committed(c,reference.snapshot(),parallel.snapshot(),continuation);++cases;
        }
        }
        c.require(routes_changed,"signed fixtures exercise changing route IDs");
        c.require(repeated_ids,"multi-token fixtures exercise repeated expert IDs");
        std::printf("\n%s whole-GDN engineering parity cases=%d continuation-cases=%d failures=%d; no full-model inference/quality/performance claim\n",c.failures?"FAIL":"PASS",cases,cases,c.failures);
        if(c.failures)return 1;
        if(o.benchmark){benchmark(c,o,reference,parallel,epoch,state,conv);std::printf("BENCH_GATE PASS all measured pairs and warmups passed parity; one-layer wall times only\n");}
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"tp2_gdn_layer: %s\n",e.what());return 2;}
}
