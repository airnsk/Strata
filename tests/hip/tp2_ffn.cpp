// Test-only, routed-expert FFN TP2 proof. No production path or shared expert.
// Requires two gfx906 devices with bidirectional peer access. All timings include
// host orchestration and compact input/result transfers, but not weight uploads.
#include "strata/core/tp2_shard.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "tp2_ffn_join.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace tp = strata::core::tp2;
namespace {
constexpr int N = 2560, F = 640;
// Predeclared gross-error guards, NOT full-model quality acceptance thresholds.
constexpr double max_scaled_limit = 3e-3, rms_relative_limit = 1e-3;
constexpr double norm_floor = 1e-12, component_atol = 1e-5, component_rtol = 3e-3;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void on(int device) { ck(cudaSetDevice(device), "set device"); }
template<class T> T* alloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, std::max<size_t>(n,1) * sizeof(T)), "allocate");
    return p;
}
template<class T> void upload(T* p, const std::vector<T>& v, cudaStream_t s) {
    if (!v.empty()) ck(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s),"upload");
}
template<class T> std::vector<T> download(const T* p, size_t n) {
    std::vector<T> out(n);
    if (n) ck(cudaMemcpy(out.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"download");
    return out;
}
enum class Exchange { Rows, ReducedHost, Reduced };
struct Options {
    Exchange exchange = Exchange::Reduced;
    int iterations = 12, mode = 8, experts = 10, layer = 0;
    bool graphs = true;
    std::vector<int> tokens{1,2,4,8};
    std::vector<std::string> gguf;
};
int integer(const std::string& s) {
    size_t end = 0;
    int v = std::stoi(s,&end);
    if (end != s.size()) throw std::invalid_argument("not an integer: " + s);
    return v;
}
Options options(int argc, char** argv) {
    Options o;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        auto value = [&]() -> std::string {
            if (++a >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[a];
        };
        if (arg == "--iters") o.iterations = integer(value());
        else if (arg == "--mode") o.mode = integer(value());
        else if (arg == "--experts") o.experts = integer(value());
        else if (arg == "--tokens") o.tokens = {integer(value())};
        else if (arg == "--layer") o.layer = integer(value());
        else if (arg == "--gguf") o.gguf.push_back(value());
        else if (arg == "--exchange") {
            const auto v=value();
            if(v=="rows")o.exchange=Exchange::Rows;
            else if(v=="reduced-host")o.exchange=Exchange::ReducedHost;
            else if(v=="reduced")o.exchange=Exchange::Reduced;
            else throw std::invalid_argument("exchange must be rows, reduced-host or reduced");
        }
        else if (arg == "--no-graphs") o.graphs = false;
        else throw std::invalid_argument("usage: tp2_ffn [--iters 12] [--mode 8|7] [--experts 10] [--tokens 1..8] [--no-graphs] [--exchange reduced|reduced-host|rows] [--gguf SHARD ... --layer 0]");
    }
    if (o.iterations < 1 || o.iterations > 10000 || o.experts < 4 || o.experts > 16 ||
        (o.mode != 7 && o.mode != 8) || o.layer < 0 || o.layer >= 48 ||
        o.tokens[0] < 1 || o.tokens[0] > 8) throw std::invalid_argument("argument out of range");
    return o;
}
struct Fixture {
    int gu = 0, down = 0;
    std::vector<std::vector<uint8_t>> blobs;
};
std::vector<uint8_t> random_rows(int type, int rows, int cols, std::mt19937& rng) {
    const int block = type == 20 ? 32 : type == 42 ? 64 : 256;
    const size_t rb = k::iq_row_bytes(type,cols), bs = k::iq_row_bytes(type,block);
    std::vector<uint8_t> out((size_t)rows*rb);
    for (auto& x : out) x = (uint8_t)rng();
    // Actual formats all carry the fp16 block scale at byte zero. Small finite
    // nonzero scales avoid SwiGLU overflow while retaining signed nonzero outputs.
    for (size_t i = 0; i < out.size(); i += bs) {
        const uint16_t scale = (uint16_t)(0x2000u | (rng() & 0x03ffu));
        std::memcpy(out.data()+i,&scale,2);
    }
    return out;
}
Fixture synthetic(int gu, int down, int experts) {
    Fixture f{gu,down,{}};
    std::mt19937 rng((unsigned)(721+gu*13+down));
    for (int e = 0; e < experts; ++e) {
        auto b = random_rows(gu,2*F,N,rng);
        auto d = random_rows(down,N,F,rng);
        b.insert(b.end(),d.begin(),d.end());
        f.blobs.push_back(std::move(b));
    }
    return f;
}
Fixture real_weights(const Options& o) {
    // Explicit shards only: do not accidentally open/repack a whole model or
    // assume a native-dense replacement shard carries the expert tensors.
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    for (const auto& p : o.gguf) files.push_back(std::make_unique<strata::GgufFile>(p));
    const char* role[3] = {"gate","up","down"};
    const strata::TensorInfo* tensor[3]{};
    const strata::GgufFile* owner[3]{};
    for (int r = 0; r < 3; ++r) {
        const std::string name = "blk."+std::to_string(o.layer)+".ffn_"+role[r]+"_exps.weight";
        for (const auto& g : files) if (const auto* t = g->find(name)) {
            if (tensor[r]) throw std::runtime_error("duplicate tensor " + name);
            tensor[r] = t; owner[r] = g.get();
        }
        if (!tensor[r]) throw std::runtime_error("missing " + name + "; supply its --gguf shard");
        const auto& shape = tensor[r]->shape;
        if (shape.size()!=3 || shape[0]!=(uint64_t)(r==2?F:N) || shape[1]!=(uint64_t)(r==2?N:F) ||
            shape[2]<(uint64_t)o.experts) throw std::runtime_error("unexpected shape: " + name);
        const uint64_t bytes = strata::tensor_payload_bytes(*tensor[r]);
        const uint64_t file = owner[r]->file_size(), base = owner[r]->data_start();
        if (!bytes || base > file || tensor[r]->offset > file-base || bytes > file-base-tensor[r]->offset)
            throw std::runtime_error("tensor is outside file: " + name);
    }
    if (tensor[0]->type != tensor[1]->type || tensor[0]->shape[2] != tensor[1]->shape[2] ||
        tensor[0]->shape[2] != tensor[2]->shape[2]) throw std::runtime_error("inconsistent expert roles");
    Fixture f{(int)tensor[0]->type,(int)tensor[2]->type,{}};
    (void)tp::plan(f.gu,f.down,N,F); // refuse unsupported source formats before copying
    const auto L = k::native_expert_layout(f.gu,f.down,N,F);
    const size_t size[3] = {L.up_off,L.up_off,L.bytes-L.down_off};
    const size_t off[3] = {0,L.up_off,L.down_off};
    for (int e = 0; e < o.experts; ++e) {
        // Widely spaced distinct source experts, in range for pruned models too.
        const size_t id = (size_t)e * (size_t)tensor[0]->shape[2] / (size_t)o.experts;
        std::vector<uint8_t> b(L.bytes);
        for (int r = 0; r < 3; ++r)
            std::memcpy(b.data()+off[r],owner[r]->tensor_data(*tensor[r])+id*size[r],size[r]);
        f.blobs.push_back(std::move(b));
    }
    std::printf("source=real layer=%d selected=%d distinct experts (mmap; only selected payload copied)\n",o.layer,o.experts);
    return f;
}
struct Rank {
    int device, T, K, count, first, ff;
    k::NativeExpertLayout L;
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    cudaEvent_t compute_start{},compute_end{};
    std::vector<uint8_t*> blobs;
    float *x=nullptr,*out=nullptr,*local_sum=nullptr,*route_weights=nullptr;
    uint8_t *xq=nullptr,*scratch=nullptr,*record=nullptr,*host_record=nullptr;
    size_t record_bytes=0,weights_offset=0;
    bool packed=false,static_ready=false;
    unsigned long long* pointers=nullptr;
    int32_t *starts=nullptr,*groups=nullptr,*dst=nullptr,*tok=nullptr;
    std::vector<unsigned long long> hp;
    std::vector<int32_t> hs,ht,hd,hg;
    Rank(const Fixture& f,int dev,int tokens,int selected,int begin,int groups_n,bool tensor,bool capture,bool packed_metadata=false)
      : device(dev),T(tokens),K(selected),count(groups_n),first(begin),ff(tensor?F/2:F),
        L(k::native_expert_layout(f.gu,f.down,N,ff)),packed(packed_metadata) {
        on(device);
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
        ck(cudaEventCreate(&compute_start),"event"); ck(cudaEventCreate(&compute_end),"event");
        const auto plan = tp::plan(f.gu,f.down,N,F);
        if (count) for (const auto& original : f.blobs) {
            auto b = tensor ? tp::split(plan,original,(unsigned)device) : original;
            auto* p = alloc<uint8_t>(b.size());
            ck(cudaMemcpy(p,b.data(),b.size(),cudaMemcpyHostToDevice),"weight upload");
            blobs.push_back(p);
        }
        x=alloc<float>((size_t)T*N); xq=alloc<uint8_t>((size_t)T*(N/32)*36);
        out=alloc<float>((size_t)std::max(1,count)*T*N);
        scratch=alloc<uint8_t>(k::native_expert_scratch_bytes(std::max(1,count*T),ff));
        local_sum=alloc<float>((size_t)T*N);
        if(packed){
            weights_offset=((size_t)count*sizeof(unsigned long long)+15)&~size_t(15);
            record_bytes=weights_offset+(size_t)T*K*sizeof(float);
            record=alloc<uint8_t>(record_bytes);
            ck(cudaHostAlloc((void**)&host_record,record_bytes,cudaHostAllocPortable),"pinned dynamic metadata");
            std::memset(host_record,0,record_bytes);
            pointers=reinterpret_cast<unsigned long long*>(record);
            route_weights=reinterpret_cast<float*>(record+weights_offset);
        }else pointers=alloc<unsigned long long>(count);
        starts=alloc<int32_t>(count+1);
        groups=alloc<int32_t>(1); dst=alloc<int32_t>(count*T); tok=alloc<int32_t>(count*T);
        hp.resize(count); hs.resize(count+1); hg={count}; ht.resize(count*T); hd.resize(count*T);
        for (int g=0;g<count;++g) {
            hs[g]=g*T;
            for (int t=0;t<T;++t) { ht[g*T+t]=t; hd[g*T+t]=t*count+g; }
        }
        hs[count]=count*T;
        metadata(0);
        ck(cudaStreamSynchronize(stream),"initial metadata");
        if (capture && count) {
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"capture begin");
            compute();
            ck(cudaStreamEndCapture(stream,&graph),"capture end");
            ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0),"instantiate");
        }
    }
    Rank(const Rank&)=delete;
    ~Rank() {
        cudaSetDevice(device);
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        for(auto p:blobs) cudaFree(p);
        cudaFree(x);cudaFree(xq);cudaFree(out);cudaFree(scratch);cudaFree(local_sum);
        if(packed){cudaFree(record);cudaFreeHost(host_record);}else cudaFree(pointers);
        cudaFree(starts);cudaFree(groups);cudaFree(dst);cudaFree(tok);
        if(compute_start)cudaEventDestroy(compute_start);
        if(compute_end)cudaEventDestroy(compute_end);
        if(stream)cudaStreamDestroy(stream);
    }
    void metadata(int epoch,const std::vector<float>* weights=nullptr) {
        on(device);
        // Route rotation exercises live pointer metadata inside captured graphs.
        // EP ownership counts are controlled synthetic assignments, not a replay
        // of the production hot-profile cache. Both paths select the same K blobs.
        for(int g=0;g<count;++g) hp[g]=(unsigned long long)blobs[(first+g+epoch)%K];
        if(packed){
            if(!hp.empty())std::memcpy(host_record,hp.data(),hp.size()*sizeof(hp[0]));
            if(weights)std::memcpy(host_record+weights_offset,weights->data(),(size_t)T*K*sizeof(float));
            ck(cudaMemcpyAsync(record,host_record,record_bytes,cudaMemcpyHostToDevice,stream),"live pinned route record");
        }else upload(pointers,hp,stream);
        if(!packed || !static_ready){
            upload(starts,hs,stream);upload(groups,hg,stream);upload(dst,hd,stream);upload(tok,ht,stream);
            static_ready=true;
        }
    }
    void compute() {
        if (!count) return;
        k::quantize_q8_1_rows(x,T,N,xq,stream);
        k::native_expert_grouped(L,pointers,starts,groups,dst,tok,count,count*T,xq,scratch,out,stream);
    }
    void poison() {
        on(device);
        ck(cudaMemsetAsync(out,0xff,(size_t)std::max(1,count)*T*N*sizeof(float),stream),"poison output");
        ck(cudaMemsetAsync(local_sum,0xff,(size_t)T*N*sizeof(float),stream),"poison reduced output");
    }
    void launch(bool poison_now=true) {
        on(device);
        if(poison_now)poison();
        ck(cudaEventRecord(compute_start,stream),"compute start");
        if(executable) ck(cudaGraphLaunch(executable,stream),"graph launch");
        else compute();
        ck(cudaEventRecord(compute_end,stream),"compute end");
    }
    void reduce(float* destination,bool peer) {
        on(device);tp2_rank_reduce(out,route_weights,destination,T,K,N,count,first,peer,stream);
    }
    void sync() { on(device);ck(cudaStreamSynchronize(stream),"rank sync"); }
    std::vector<uint8_t> hidden() {
        on(device);
        const size_t f=(size_t)count*T*ff*sizeof(float), aligned=(f+255)&~size_t(255);
        return download(scratch+3*aligned,(size_t)count*T*(ff/32)*36);
    }
    double compute_us() { on(device);float ms=0;ck(cudaEventElapsedTime(&ms,compute_start,compute_end),"compute elapsed");return ms*1000.; }
};
struct Results { std::vector<float> rows,combined; std::vector<uint8_t> hq; };
struct Join {
    int T,K;
    float *peer=nullptr,*weights=nullptr,*rows=nullptr,*out=nullptr;
    cudaEvent_t transfer_start{},transfer_end{};
    Join(int t,int k):T(t),K(k) {
        on(0);peer=alloc<float>((size_t)T*K*N);weights=alloc<float>(T*K);
        rows=alloc<float>((size_t)T*K*N);out=alloc<float>((size_t)T*N);
        ck(cudaEventCreate(&transfer_start),"join event");ck(cudaEventCreate(&transfer_end),"join event");
    }
    ~Join(){cudaSetDevice(0);cudaFree(peer);cudaFree(weights);cudaFree(rows);cudaFree(out);
        cudaEventDestroy(transfer_start);cudaEventDestroy(transfer_end);}
    void run(Rank& a,Rank* b,const std::vector<float>& w,bool tp) {
        // No device-side spin. Complete peer compute explicitly, then return its
        // compact rows on primary's stream after primary compute. Wall time counts it.
        if(b)b->sync();
        on(0);upload(weights,w,a.stream);
        ck(cudaEventRecord(transfer_start,a.stream),"return start");
        if(b && b->count) ck(cudaMemcpyPeerAsync(peer,0,b->out,1,(size_t)b->count*T*N*sizeof(float),a.stream),"return P2P");
        ck(cudaEventRecord(transfer_end,a.stream),"return end");
        tp2_ffn_join(a.out,peer,weights,rows,out,T,K,N,a.count,tp,a.stream);
        a.sync();
    }
    Results result() {on(0);return {download(rows,(size_t)T*K*N),download(out,(size_t)T*N),{}};}
    double transfer_us(){on(0);float ms=0;ck(cudaEventElapsedTime(&ms,transfer_start,transfer_end),"return elapsed");return ms*1000.;}
};
std::vector<float> input(int T,int epoch) {
    std::vector<float> x((size_t)T*N);
    for(size_t i=0;i<x.size();++i) x[i]=0.13f*std::sin((float)(i*7+epoch*113)*0.031f)+0.07f*std::cos((float)(i+epoch*61)*0.017f);
    return x;
}
std::vector<float> routing(int T,int K,int epoch) {
    std::vector<float> w(T*K);
    for(int t=0;t<T;++t){float sum=0;for(int e=0;e<K;++e){w[t*K+e]=(float)(1+(e*7+t*3+epoch*5)%17);sum+=w[t*K+e];}
        for(int e=0;e<K;++e)w[t*K+e]/=sum;}
    if(epoch==1){std::fill(w.begin(),w.end(),0.0f);for(int t=0;t<T;++t)w[t*K+(t+1)%K]=1.0f;}
    if(epoch==2)std::fill(w.begin(),w.end(),0.0f);
    // Signed weights are a numerical cancellation stress fixture, not real router semantics.
    if(epoch==3)for(int t=0;t<T;++t)for(int e=0;e<K;++e)w[t*K+e]=(e%2?-1.f:1.f)/(float)K;
    return w;
}
struct Error {double max_abs=0,max_ref=0,sq=0,ref_sq=0;size_t bits=0,component_fail=0;bool finite=true;};
bool compare(const std::vector<float>& ref,const std::vector<float>& got,const char* label,bool exact,bool verbose=true,bool allow_zero=false) {
    if(ref.size()!=got.size())throw std::runtime_error("comparison size mismatch");
    Error e;
    for(size_t i=0;i<ref.size();++i){
        if(!std::isfinite(ref[i])||!std::isfinite(got[i]))e.finite=false;
        const double d=std::fabs((double)got[i]-ref[i]),r=std::fabs((double)ref[i]);
        e.max_abs=std::max(e.max_abs,d);e.max_ref=std::max(e.max_ref,r);e.sq+=d*d;e.ref_sq+=(double)ref[i]*ref[i];
        e.bits+=std::memcmp(&ref[i],&got[i],sizeof(float))!=0;
        e.component_fail+=d>component_atol+component_rtol*r;
    }
    const double scaled=e.max_abs/std::max(e.max_ref,norm_floor);
    const double rms=std::sqrt(e.sq/std::max(e.ref_sq,norm_floor*norm_floor));
    const bool nontrivial=e.max_ref>norm_floor;
    const bool ok=e.finite && (nontrivial || (allow_zero && e.max_abs==0)) &&
        (exact ? e.bits==0 : scaled<=max_scaled_limit&&rms<=rms_relative_limit);
    if(verbose || !ok)std::printf("  %s %s max_abs=%.9g max_ref=%.9g max_scaled=%.9g rms_rel=%.9g bitdiff=%zu component_diag=%zu\n",label,ok?"PASS":"FAIL",e.max_abs,e.max_ref,scaled,rms,e.bits,e.component_fail);
    return ok;
}
bool hidden_exact(const std::vector<uint8_t>& full,const std::vector<uint8_t>& a,const std::vector<uint8_t>& b,int entries,bool verbose=true) {
    const size_t half=(F/2/32)*36;
    if(full.size()!=(size_t)entries*2*half||a.size()!=(size_t)entries*half||b.size()!=a.size())throw std::runtime_error("hidden size");
    size_t diff=0;
    for(int e=0;e<entries;++e) for(size_t j=0;j<half;++j){
        diff+=full[(size_t)e*2*half+j]!=a[(size_t)e*half+j];
        diff+=full[(size_t)e*2*half+half+j]!=b[(size_t)e*half+j];
    }
    if(verbose || diff)std::printf("  hidden_q8_blocks %s differing_bytes=%zu\n",diff?"FAIL":"PASS",diff);
    return diff==0;
}
bool peer_ready(Exchange exchange) {
    int count=0;auto e=cudaGetDeviceCount(&count);
    if(e!=cudaSuccess||count<2){std::printf("SKIP: two GPUs required\n");return false;}
    for(int d=0;d<2;++d){
        cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,d),"device properties");
        std::printf("device%d=%s CUs=%d total_GiB=%.3f\n",d,p.name,p.multiProcessorCount,(double)p.totalGlobalMem/(1ull<<30));
#if defined(STRATA_HIP_GFX906)
        if(std::strncmp(p.gcnArchName,"gfx906",6)!=0){
            std::printf("SKIP: this prototype requires gfx906; device%d architecture=%s\n",d,p.gcnArchName);return false;
        }
#endif
        int can=0;ck(cudaDeviceCanAccessPeer(&can,d,1-d),"peer capability");
        if(!can){std::printf("SKIP: bidirectional peer access required; no host fallback in this harness\n");return false;}
        on(d);e=cudaDeviceEnablePeerAccess(1-d,0);
        if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(e,"enable peer access");
    }
    std::array<uint8_t*,2> p{};
    std::vector<uint8_t> pattern(4096);
    for(size_t i=0;i<pattern.size();++i)pattern[i]=(uint8_t)(i*73+19);
    for(int d=0;d<2;++d){on(d);p[d]=alloc<uint8_t>(pattern.size());}
    for(int src=0;src<2;++src){
        on(src);ck(cudaMemcpy(p[src],pattern.data(),pattern.size(),cudaMemcpyHostToDevice),"peer probe source");
        on(1-src);ck(cudaMemset(p[1-src],0,pattern.size()),"peer probe clear");
        cudaStream_t s{};ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"probe stream");
        ck(cudaMemcpyPeerAsync(p[1-src],1-src,p[src],src,pattern.size(),s),"peer probe copy");
        ck(cudaStreamSynchronize(s),"peer probe sync");
        if(download(p[1-src],pattern.size())!=pattern)throw std::runtime_error("peer copy content mismatch");
        ck(cudaStreamDestroy(s),"probe destroy");
    }
    if(exchange!=Exchange::Rows)for(int src=0;src<2;++src){
        cudaStream_t producer{},consumer{};cudaEvent_t ready{};
        on(src);ck(cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking),"push probe producer");
        ck(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming),"push probe event");
        on(1-src);ck(cudaStreamCreateWithFlags(&consumer,cudaStreamNonBlocking),"push probe consumer");
        for(unsigned sequence=1;sequence<=16;++sequence){
            // Monotonic sequences exceed the fixture replay period, exposing stale event reuse.
            for(size_t i=0;i<pattern.size();++i)pattern[i]=(uint8_t)(i*73+sequence*29+(i>>8)*sequence);
            on(src);ck(cudaMemcpy(p[src],pattern.data(),pattern.size(),cudaMemcpyHostToDevice),"push probe source");
            on(1-src);ck(cudaMemset(p[1-src],0,pattern.size()),"push probe poison");
            on(src);tp2_input_push(reinterpret_cast<float*>(p[src]),reinterpret_cast<float*>(p[1-src]),(int)pattern.size()/4,producer);
            ck(cudaEventRecord(ready,producer),"push probe record");
            on(1-src);ck(cudaStreamWaitEvent(consumer,ready,0),"push probe cross-device wait");
            ck(cudaStreamSynchronize(consumer),"push probe consumer done");
            if(download(p[1-src],pattern.size())!=pattern)throw std::runtime_error("peer-write/event visibility mismatch at sequence "+std::to_string(sequence));
        }
        on(1-src);cudaStreamDestroy(consumer);on(src);cudaStreamDestroy(producer);cudaEventDestroy(ready);
    }
    for(int d=0;d<2;++d){on(d);cudaFree(p[d]);}
    if(exchange==Exchange::Rows)std::printf("transport probe: runtime peer copies passed content checks both directions; no physical DMA bandwidth claim\n");
    else std::printf("transport probe: runtime peer copies and direct peer writes+system fences+cross-device events passed content checks both directions; 16 monotonic sequences/direction; no physical DMA bandwidth claim\n");
    return true;
}
struct ExchangeEvents {
    cudaEvent_t input_start{},input_end{},input_ready{},output_start{},output_end{},peer_ready{};
    ExchangeEvents(){
        on(0);ck(cudaEventCreate(&input_start),"input event");ck(cudaEventCreate(&input_end),"input event");
        ck(cudaEventCreateWithFlags(&input_ready,cudaEventDisableTiming),"input dependency");
        on(1);ck(cudaEventCreate(&output_start),"output event");ck(cudaEventCreate(&output_end),"output event");
        ck(cudaEventCreateWithFlags(&peer_ready,cudaEventDisableTiming),"peer dependency");
    }
    ~ExchangeEvents(){cudaSetDevice(0);cudaEventDestroy(input_start);cudaEventDestroy(input_end);cudaEventDestroy(input_ready);
        cudaSetDevice(1);cudaEventDestroy(output_start);cudaEventDestroy(output_end);cudaEventDestroy(peer_ready);}
    double input_us(){on(0);float ms=0;ck(cudaEventElapsedTime(&ms,input_start,input_end),"input elapsed");return ms*1000.;}
    double output_us(){on(1);float ms=0;ck(cudaEventElapsedTime(&ms,output_start,output_end),"reduce/push elapsed");return ms*1000.;}
};
void run_reduced(Rank& a,Rank& b,Join& join,const std::vector<float>& w,int epoch,Exchange exchange,ExchangeEvents& ev){
    // Live routes and weights remain inside wall timing; only invariant tables
    // were uploaded during construction. Pinned buffers remain alive until DAG completion.
    a.metadata(epoch,&w);b.metadata(epoch,&w);
    on(0);ck(cudaEventRecord(ev.input_start,a.stream),"input push start");
    if(b.count)tp2_input_push(a.x,b.x,a.T*N,a.stream);
    ck(cudaEventRecord(ev.input_end,a.stream),"input push end");
    ck(cudaEventRecord(ev.input_ready,a.stream),"input ready");
    a.launch(false);a.reduce(a.local_sum,false);
    on(1);ck(cudaStreamWaitEvent(b.stream,ev.input_ready,0),"peer waits input");
    b.launch(false);
    ck(cudaEventRecord(ev.output_start,b.stream),"reduce/push start");
    b.reduce(exchange==Exchange::Reduced?join.peer:b.local_sum,exchange==Exchange::Reduced);
    ck(cudaEventRecord(ev.output_end,b.stream),"reduce/push end");
    ck(cudaEventRecord(ev.peer_ready,b.stream),"peer output ready");
    if(exchange==Exchange::ReducedHost){
        // Attribution control: reduced payload, but original host join/runtime return copy.
        b.sync();on(0);
        ck(cudaEventRecord(join.transfer_start,a.stream),"runtime return start");
        ck(cudaMemcpyPeerAsync(join.peer,0,b.local_sum,1,(size_t)a.T*N*sizeof(float),a.stream),"reduced runtime return");
        ck(cudaEventRecord(join.transfer_end,a.stream),"runtime return end");
    }else{
        on(0);ck(cudaStreamWaitEvent(a.stream,ev.peer_ready,0),"primary waits peer output");
    }
    tp2_sum_vectors(a.local_sum,join.peer,join.out,a.T*N,a.stream);
    a.sync(); // sole host completion wait for the event-joined mode
}
Results reduced_result(Rank& a,Rank& b,Join& join,bool tp){
    on(0);auto left=download(a.out,(size_t)a.count*a.T*N);
    auto combined=download(join.out,(size_t)a.T*N);
    on(1);auto right=download(b.out,(size_t)b.count*b.T*N);
    std::vector<float> rows((size_t)a.T*a.K*N);
    for(int t=0;t<a.T;++t)for(int e=0;e<a.K;++e)for(int j=0;j<N;++j){
        const size_t dst=((size_t)t*a.K+e)*N+j;
        if(tp)rows[dst]=left[dst]+right[dst];
        else if(e<a.count)rows[dst]=left[((size_t)t*a.count+e)*N+j];
        else rows[dst]=right[((size_t)t*b.count+e-a.count)*N+j];
    }
    return {std::move(rows),std::move(combined),{}};
}
struct Timing {double wall=0,input=0,output=0,runtime_return=0,rank0=0,rank1=0;};
bool run_case(const Fixture& f,int T,const Options& o) {
    const int K=o.experts,epochs=4;
    const bool reduced=o.exchange!=Exchange::Rows;
    const char* exchange=o.exchange==Exchange::Rows?"rows":o.exchange==Exchange::ReducedHost?"reduced-host":"reduced";
    std::printf("CASE GU=%s(%d) DOWN=%s(%d) N=%d F=%d T=%d K=%d mode=%d GU=%s DOWN=LDS graph=%d exchange=%s\n",
        strata::ggml_type_name(f.gu),f.gu,strata::ggml_type_name(f.down),f.down,N,F,T,K,o.mode,
        o.mode==8?"fused-LDS-GU-SwiGLU-q8":"LDS-GU then SwiGLU/q8",(int)o.graphs,exchange);
    std::array<Results,4> reference;
    {
        Rank oracle(f,0,T,K,0,K,false,o.graphs,reduced);Join join(T,K);
        double wall=0,compute=0;
        for(int iteration=-epochs;iteration<o.iterations;++iteration){
            const int epoch=(iteration+epochs)%epochs;
            on(0);auto x=input(T,epoch);auto w=routing(T,K,epoch);
            upload(oracle.x,x,oracle.stream);if(reduced)oracle.poison();oracle.sync();
            const auto begin=std::chrono::steady_clock::now();
            oracle.metadata(epoch,reduced?&w:nullptr);oracle.launch(!reduced);
            if(reduced){oracle.reduce(join.out,false);oracle.sync();}
            else join.run(oracle,nullptr,w,false);
            const auto end=std::chrono::steady_clock::now();
            if(iteration>=0){wall+=std::chrono::duration<double,std::micro>(end-begin).count();compute+=oracle.compute_us();}
            Results result;
            if(reduced){
                on(0);const auto reduced_output=download(join.out,(size_t)T*N);
                // Independent phase-1 combine remains the oracle. The timed
                // single-GPU reduction must match it bitwise, outside timing.
                join.run(oracle,nullptr,w,false);
                result=join.result();
                if(!compare(result.combined,reduced_output,"single_reduce_vs_old_join",true,iteration<0,epoch==2))return false;
            }else result=join.result();
            if(iteration<0){reference[epoch]=std::move(result);reference[epoch].hq=oracle.hidden();}
            else if(!compare(reference[epoch].rows,result.rows,"oracle_replay_rows",true,false) ||
                    !compare(reference[epoch].combined,result.combined,"oracle_replay_combine",true,false,epoch==2))return false;
        }
        std::printf(" TIMING single-GPU-full T=%d count=%d wall_us=%.3f compute0_us=%.3f weight_bytes=%zu input_bytes=0 return_bytes=0\n",
            T,o.iterations,wall/o.iterations,compute/o.iterations,oracle.blobs.size()*oracle.L.bytes);
    }
    bool ok=true;
    const std::vector<std::pair<std::string,int>> modes={{"EP-balanced",K/2},{"EP-primary-heavy",(3*K+2)/4},
        {"EP-peer-heavy",K-(3*K+2)/4},{"EP-primary-only",K},{"EP-peer-only",0},{"TP2",-1}};
    for(const auto& [name,g0]:modes){
        const bool tp_mode=g0<0;
        Rank a(f,0,T,K,0,tp_mode?K:g0,tp_mode,o.graphs,reduced);
        Rank b(f,1,T,K,tp_mode?0:g0,tp_mode?K:K-g0,tp_mode,o.graphs,reduced);
        Join join(T,K);ExchangeEvents ev;
        on(1);cudaEvent_t in0{},in1{};ck(cudaEventCreate(&in0),"input event");ck(cudaEventCreate(&in1),"input event");
        Timing total;int measured=0;
        std::printf(" MODE %s computed_experts=%d/%d local_width=%d/%d fixture_weight_bytes=%zu/%zu dynamic_record_bytes=%zu/%zu\n",name.c_str(),a.count,b.count,a.ff,b.ff,
            a.blobs.size()*a.L.bytes,b.blobs.size()*b.L.bytes,a.record_bytes,b.record_bytes);
        for(int iteration=-epochs;iteration<o.iterations;++iteration){
            const int epoch=(iteration+epochs)%epochs;
            auto x=input(T,epoch);auto w=routing(T,K,epoch);
            on(0);upload(a.x,x,a.stream);
            if(reduced){
                a.poison();b.poison();
                on(1);ck(cudaMemsetAsync(b.x,0xff,(size_t)T*N*sizeof(float),b.stream),"poison peer input");b.sync();
                on(0);ck(cudaMemsetAsync(join.peer,0xff,(size_t)T*N*sizeof(float),a.stream),"poison peer reduced result");
            }
            a.sync(); // input resident; diagnostic poison is outside measured protocol
            const auto begin=std::chrono::steady_clock::now();
            if(reduced)run_reduced(a,b,join,w,epoch,o.exchange,ev);
            else{
                b.metadata(epoch);
                on(1);ck(cudaEventRecord(in0,b.stream),"input start");
                if(b.count)ck(cudaMemcpyPeerAsync(b.x,1,a.x,0,(size_t)T*N*sizeof(float),b.stream),"input P2P");
                ck(cudaEventRecord(in1,b.stream),"input end");b.launch();
                a.metadata(epoch);a.launch();join.run(a,&b,w,tp_mode);
            }
            const auto end=std::chrono::steady_clock::now();
            if(iteration>=0){
                ++measured;total.wall+=std::chrono::duration<double,std::micro>(end-begin).count();
                if(reduced){total.input+=ev.input_us();total.output+=ev.output_us();if(o.exchange==Exchange::ReducedHost)total.runtime_return+=join.transfer_us();}
                else{on(1);float ms=0;ck(cudaEventElapsedTime(&ms,in0,in1),"input elapsed");total.input+=ms*1000.;total.output+=join.transfer_us();}
                total.rank0+=a.compute_us();total.rank1+=b.compute_us();
            }
            // Diagnostics retain per-expert rows, but never transfer them in the timed reduced protocol.
            const auto got=reduced?reduced_result(a,b,join,tp_mode):join.result();
            const bool rows_ok=compare(reference[epoch].rows,got.rows,"expert_rows",!tp_mode,iteration<0);
            const bool combined_ok=compare(reference[epoch].combined,got.combined,"test_FP32_combine",!reduced&&!tp_mode,iteration<0,epoch==2);
            ok=rows_ok&&combined_ok&&ok;
            if(reduced && b.count){on(1);const auto received=download(b.x,x.size());if(std::memcmp(received.data(),x.data(),x.size()*sizeof(float)))throw std::runtime_error("peer input content mismatch");}
            if(tp_mode)ok=hidden_exact(reference[epoch].hq,a.hidden(),b.hidden(),T*K,iteration<0)&&ok;
            if(!ok)break;
        }
        on(1);cudaEventDestroy(in0);cudaEventDestroy(in1);
        if(measured)std::printf(" TIMING %s T=%d count=%d wall_us=%.3f input_stage_us=%.3f return_stage_us=%.3f runtime_return_copy_us=%.3f compute0_us=%.3f compute1_us=%.3f input_bytes=%zu return_bytes=%zu exchange=%s (stages are not summed; reduced-host return_stage excludes later runtime copy)\n",
            name.c_str(),T,measured,total.wall/measured,total.input/measured,total.output/measured,total.runtime_return/measured,total.rank0/measured,total.rank1/measured,
            b.count?(size_t)T*N*sizeof(float):0,reduced?(size_t)T*N*sizeof(float):(size_t)b.count*T*N*sizeof(float),exchange);
        if(!ok)return false;
    }
    return ok;
}
}
int main(int argc,char** argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    try{
        const auto o=options(argc,argv);
        // These two full-width-only experimental branches break the phase oracle.
        // Pin process-local benchmark choices; never modify the user's config.
        setenv("STRATA_EXPERT_V2","0",1);setenv("STRATA_EXPERT_V2K","0",1);
        k::native_expert_set_mode(o.mode,0);k::native_grouped_set_v1(false);k::iq_set_old_kernels(false);
        std::printf("tp2_ffn: routed experts only; no shared FFN, router, HC/KV/MTP or engine integration.\n");
        std::printf("Isolated P2P EP topology baseline, NOT production PeerExperts timing. Selected expert fixtures only; weights uploaded outside timing.\n");
        std::printf("EP fixtures replicate the small K-expert set on active ranks to rotate assignment; compute only assigned experts. This is not a model residency estimate.\n");
        std::printf("Routing coverage: four live route/input epochs (ordinary, concentrated, zero, signed-cancellation weights); T entries per expert, same selected set across tokens. No arbitrary sparse per-token routing/shared FFN.\n");
        std::printf("mode=%d (explicit), old_iq=0 grouped_v1=0 expert_v2=0 expert_v2k=0; iterations=%d\n",o.mode,o.iterations);
        std::printf("Predeclared gross-error guard: max_abs/max_ref<=%.3g and RMS_relative<=%.3g, floor=%.3g; componentwise atol=%.3g rtol=%.3g diagnostic only; hidden q8 blocks must be byte-exact. Deliberate zero-weight fixture must return exact zero; expert rows remain nonzero. Not quality approval.\n",max_scaled_limit,rms_relative_limit,norm_floor,component_atol,component_rtol);
        if(!peer_ready(o.exchange))return 77;
        int cases=0;
        if(!o.gguf.empty()){
            const auto f=real_weights(o);
            for(int t:o.tokens){if(!run_case(f,t,o))return 1;++cases;}
        }else for(int gu:{18,21,22,23})for(int down:{20,42}){
            const auto f=synthetic(gu,down,o.experts);
            for(int t:o.tokens){if(!run_case(f,t,o))return 1;++cases;}
        }
        std::printf("tp2_ffn: PASS %d cases; GPU kernel/subgraph gate only, no whole-decode result\n",cases);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"tp2_ffn: ERROR: %s\n",e.what());return 2;}
}
