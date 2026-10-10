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
    bool graphs = true, profile_stages = false;
    std::string tp_layout = "both", order = "alternate";
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
        else if (arg == "--tp-layout") o.tp_layout = value();
        else if (arg == "--order") o.order = value();
        else if (arg == "--profile-stages") o.profile_stages = true;
        else if (arg == "--no-graphs") o.graphs = false;
        else throw std::invalid_argument("usage: tp2_ffn [--iters 12] [--mode 8|7] [--experts 10] [--tokens 1..8] [--no-graphs] [--exchange reduced|reduced-host|rows] [--tp-layout column|row|both] [--order alternate|forward|reverse] [--profile-stages] [--gguf SHARD ... --layer 0]");
    }
    if (o.iterations < 1 || o.iterations > 10000 || o.experts < 4 || o.experts > 16 ||
        (o.mode != 7 && o.mode != 8) || o.layer < 0 || o.layer >= 48 ||
        o.tokens[0] < 1 || o.tokens[0] > 8) throw std::invalid_argument("argument out of range");
    if (o.tp_layout != "column" && o.tp_layout != "row" && o.tp_layout != "both")
        throw std::invalid_argument("tp-layout must be column, row or both");
    if (o.order != "alternate" && o.order != "forward" && o.order != "reverse")
        throw std::invalid_argument("order must be alternate, forward or reverse");
    if (o.tp_layout != "column" && o.exchange != Exchange::Reduced)
        throw std::invalid_argument("row/both TP requires --exchange reduced; use --tp-layout column for legacy transport controls");
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
    int device, T, K, count, first, ff, width, mode;
    bool row_tp;
    k::NativeExpertLayout L, down_L;
    cudaStream_t stream{};
    cudaGraph_t graph{}, down_graph{};
    cudaGraphExec_t executable{}, down_executable{};
    cudaEvent_t compute_start{},compute_end{};
    std::vector<uint8_t*> blobs;
    float *x=nullptr,*out=nullptr,*local_sum=nullptr,*route_weights=nullptr;
    uint8_t *xq=nullptr,*scratch=nullptr,*record=nullptr,*host_record=nullptr;
    size_t record_bytes=0,weights_offset=0;
    uint8_t* down_scratch=nullptr;
    bool packed=false,static_ready=false;
    unsigned long long* pointers=nullptr;
    int32_t *starts=nullptr,*groups=nullptr,*dst=nullptr,*tok=nullptr;
    std::vector<unsigned long long> hp;
    std::vector<int32_t> hs,ht,hd,hg;
    Rank(const Fixture& f,int dev,int tokens,int selected,int begin,int groups_n,bool tensor,bool capture,bool packed_metadata=false, bool output_rows=false, int kernel_mode=8)
      : device(dev),T(tokens),K(selected),count(groups_n),first(begin),ff(tensor?F/2:F),
        width(output_rows?N/2:N),mode(kernel_mode),row_tp(output_rows),
        L(k::native_expert_layout(f.gu,f.down,N,ff)),packed(packed_metadata) {
        on(device);
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
        ck(cudaEventCreate(&compute_start),"event"); ck(cudaEventCreate(&compute_end),"event");
        const auto plan = tp::plan(f.gu,f.down,N,F,row_tp?tp::DownSplit::OutputRows:tp::DownSplit::Columns);
        if(row_tp){
            // GU retains N input and F/2 rows; down uses F input and N/2 rows.
            // Both views address the SAME unchanged native-block shard blob.
            L.bytes=plan.shard.bytes;
            down_L=k::native_expert_layout(f.gu,f.down,N/2,F);
            down_L.down_off=plan.shard.down.offset;
            down_L.bytes=plan.shard.bytes;
            down_scratch=alloc<uint8_t>(k::native_expert_scratch_bytes(count*T,F));
        }
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
            if(row_tp){
                ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"down capture begin");
                compute_down();
                ck(cudaStreamEndCapture(stream,&down_graph),"down capture end");
                ck(cudaGraphInstantiate(&down_executable,down_graph,nullptr,nullptr,0),"down instantiate");
            }
        }
    }
    Rank(const Rank&)=delete;
    ~Rank() {
        cudaSetDevice(device);
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        if(down_executable) cudaGraphExecDestroy(down_executable);
        if(down_graph) cudaGraphDestroy(down_graph);
        cudaFree(down_scratch);
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
        k::native_expert_set_mode(mode,row_tp?1:0);
        k::native_expert_grouped(L,pointers,starts,groups,dst,tok,count,count*T,xq,scratch,out,stream);
        k::native_expert_set_mode(mode,0);
    }
    void compute_down() {
        k::native_expert_set_mode(mode,2);
        k::native_expert_grouped(down_L,pointers,starts,groups,dst,tok,count,count*T,xq,down_scratch,out,stream);
        k::native_expert_set_mode(mode,0);
    }
    void launch_down() {
        on(device);
        if(down_executable) ck(cudaGraphLaunch(down_executable,stream),"down graph launch");
        else compute_down();
    }
    uint8_t* hidden_ptr(bool full=false) {
        const size_t bytes=(size_t)count*T*(full?F:ff)*sizeof(float);
        return (full?down_scratch:scratch)+3*((bytes+255)&~size_t(255));
    }
    void poison() {
        on(device);
        ck(cudaMemsetAsync(out,0xff,(size_t)std::max(1,count)*T*N*sizeof(float),stream),"poison output");
        ck(cudaMemsetAsync(local_sum,0xff,(size_t)T*N*sizeof(float),stream),"poison reduced output");
    }
    void launch(bool poison_now=false, bool profile=false) {
        on(device);
        if(poison_now)poison();
        if(profile) ck(cudaEventRecord(compute_start,stream),"compute start");
        if(executable) ck(cudaGraphLaunch(executable,stream),"graph launch");
        else compute();
        if(profile) ck(cudaEventRecord(compute_end,stream),"compute end");
    }
    void reduce(float* destination,bool peer) {
        on(device);tp2_rank_reduce(out,route_weights,destination,T,K,width,count,first,peer,stream);
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
    void run(Rank& a,Rank* b,const std::vector<float>& w,bool tp,bool profile=false) {
        // No device-side spin. Complete peer compute explicitly, then return its
        // compact rows on primary's stream after primary compute. Wall time counts it.
        if(b)b->sync();
        on(0);upload(weights,w,a.stream);
        if(profile) ck(cudaEventRecord(transfer_start,a.stream),"return start");
        if(b && b->count) ck(cudaMemcpyPeerAsync(peer,0,b->out,1,(size_t)b->count*T*N*sizeof(float),a.stream),"return P2P");
        if(profile) ck(cudaEventRecord(transfer_end,a.stream),"return end");
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
void probe_pattern(std::vector<uint8_t>& bytes,unsigned source,unsigned sequence) {
    // Distinct finite FP32 words, including across sequences: no NaN payload ambiguity.
    for(size_t i=0;i<bytes.size()/4;++i){
        const uint32_t word=0x3f000000u | ((sequence*4099u+(uint32_t)i*97u+source*65537u)&0x007fffffu);
        std::memcpy(bytes.data()+i*4,&word,4);
    }
}
std::string probe_mismatch(const char* stage,int src,unsigned sequence,const std::vector<uint8_t>& expected,
                           const std::vector<uint8_t>& got) {
    if(expected.size()!=got.size())throw std::runtime_error("probe result size mismatch");
    const auto mismatch=std::mismatch(expected.begin(),expected.end(),got.begin());
    if(mismatch.first==expected.end())return {};
    const size_t at=(size_t)(mismatch.first-expected.begin()),word_at=at&~size_t(3);
    uint32_t want=0,have=0;std::memcpy(&want,expected.data()+word_at,4);std::memcpy(&have,got.data()+word_at,4);
    char detail[384];
    std::snprintf(detail,sizeof(detail),"%s src=%d dst=%d sequence=%u first_byte=%zu expected_byte=0x%02x got_byte=0x%02x word_offset=%zu expected_word=0x%08x got_word=0x%08x",
                  stage,src,1-src,sequence,at,(unsigned)expected[at],(unsigned)got[at],word_at,(unsigned)want,(unsigned)have);
    return detail;
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
    std::array<uint8_t*,2> p{},witness{};
    std::vector<uint8_t> pattern(4096);
    for(int d=0;d<2;++d){on(d);p[d]=alloc<uint8_t>(pattern.size());witness[d]=alloc<uint8_t>(pattern.size());}
    for(int src=0;src<2;++src){
        cudaStream_t producer{},consumer{};cudaEvent_t ready{};
        on(src);ck(cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking),"probe producer");
        ck(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming),"probe event");
        on(1-src);ck(cudaStreamCreateWithFlags(&consumer,cudaStreamNonBlocking),"probe consumer");
        auto initialize=[&](unsigned sequence){
            probe_pattern(pattern,(unsigned)src,sequence);
            // Explicitly finish BOTH initializations. hipMemset's default-stream
            // async work has no ordering edge to another device's nonblocking
            // producer: late poison must never race a tested peer write.
            on(src);ck(cudaMemcpyAsync(p[src],pattern.data(),pattern.size(),cudaMemcpyHostToDevice,producer),"probe source upload");
            ck(cudaStreamSynchronize(producer),"probe source initialization complete");
            on(1-src);ck(cudaMemsetAsync(p[1-src],0,pattern.size(),consumer),"probe destination poison");
            ck(cudaMemsetAsync(witness[1-src],0,pattern.size(),consumer),"probe witness poison");
            ck(cudaStreamSynchronize(consumer),"probe destination initialization complete");
        };
        initialize(0);
        on(1-src);ck(cudaMemcpyPeerAsync(p[1-src],1-src,p[src],src,pattern.size(),consumer),"runtime peer probe copy");
        ck(cudaStreamSynchronize(consumer),"runtime peer probe complete");
        if(const auto bad=probe_mismatch("runtime peer copy",src,0,pattern,download(p[1-src],pattern.size()));!bad.empty())
            throw std::runtime_error(bad);
        if(exchange!=Exchange::Rows)for(unsigned sequence=1;sequence<=16;++sequence){
            initialize(sequence);
            on(src);tp2_input_push(reinterpret_cast<float*>(p[src]),reinterpret_cast<float*>(p[1-src]),(int)pattern.size()/4,producer);
            ck(cudaEventRecord(ready,producer),"push probe record");
            on(1-src);ck(cudaStreamWaitEvent(consumer,ready,0),"push probe cross-device wait");
            // A GPU-side local witness must see the peer data after the event,
            // not only a later host/DMA readback that could mask consumer visibility.
            tp2_input_push(reinterpret_cast<float*>(p[1-src]),reinterpret_cast<float*>(witness[1-src]),(int)pattern.size()/4,consumer);
            ck(cudaStreamSynchronize(consumer),"push probe consumer complete");
            const auto bad=probe_mismatch("peer-write/event consumer witness",src,sequence,pattern,download(witness[1-src],pattern.size()));
            if(!bad.empty()){
                std::fprintf(stderr,"probe failure: %s\n",bad.c_str());
                const auto raw=probe_mismatch("destination readback after consumer wait",src,sequence,pattern,download(p[1-src],pattern.size()));
                std::fprintf(stderr,"probe diagnostic: %s\n",raw.empty()?"destination readback matches":raw.c_str());
                // One untimed fully synchronized diagnostic. Recovery does NOT
                // turn a failed event protocol into a pass or select a fallback.
                on(src);ck(cudaStreamSynchronize(producer),"diagnostic producer completion");
                on(1-src);ck(cudaStreamSynchronize(consumer),"diagnostic consumer completion");
                tp2_input_push(reinterpret_cast<float*>(p[1-src]),reinterpret_cast<float*>(witness[1-src]),(int)pattern.size()/4,consumer);
                ck(cudaStreamSynchronize(consumer),"diagnostic witness complete");
                const auto after=probe_mismatch("witness after explicit producer completion",src,sequence,pattern,download(witness[1-src],pattern.size()));
                std::fprintf(stderr,"probe diagnostic: %s; original event gate remains FAIL\n",after.empty()?"values match after producer completion":after.c_str());
                throw std::runtime_error(bad+"; original failure preserved after diagnostic");
            }
        }
        on(1-src);cudaStreamDestroy(consumer);on(src);cudaStreamDestroy(producer);cudaEventDestroy(ready);
    }
    for(int d=0;d<2;++d){on(d);cudaFree(p[d]);cudaFree(witness[d]);}
    if(exchange==Exchange::Rows)std::printf("transport probe: explicit initialization and runtime peer copies passed content checks both directions; no physical DMA bandwidth claim\n");
    else std::printf("transport probe: initialized runtime copies and direct peer writes+system fences+cross-device events+GPU consumer witness passed; 16 monotonic sequences/direction; no physical DMA bandwidth claim\n");
    return true;
}
struct ExchangeEvents {
    cudaEvent_t input_start{},input_end{},input_ready{},output_start{},output_end{},peer_ready{},hidden_ready0{},hidden_ready1{};
    ExchangeEvents(){
        on(0);ck(cudaEventCreate(&input_start),"input event");ck(cudaEventCreate(&input_end),"input event");
        ck(cudaEventCreateWithFlags(&input_ready,cudaEventDisableTiming),"input dependency");
        ck(cudaEventCreateWithFlags(&hidden_ready0,cudaEventDisableTiming),"hidden dependency0");
        on(1);ck(cudaEventCreate(&output_start),"output event");ck(cudaEventCreate(&output_end),"output event");
        ck(cudaEventCreateWithFlags(&peer_ready,cudaEventDisableTiming),"peer dependency");
        ck(cudaEventCreateWithFlags(&hidden_ready1,cudaEventDisableTiming),"hidden dependency1");
    }
    ~ExchangeEvents(){cudaSetDevice(0);cudaEventDestroy(input_start);cudaEventDestroy(input_end);cudaEventDestroy(input_ready);cudaEventDestroy(hidden_ready0);
        cudaSetDevice(1);cudaEventDestroy(output_start);cudaEventDestroy(output_end);cudaEventDestroy(peer_ready);cudaEventDestroy(hidden_ready1);}
    double input_us(){on(0);float ms=0;ck(cudaEventElapsedTime(&ms,input_start,input_end),"input elapsed");return ms*1000.;}
    double output_us(){on(1);float ms=0;ck(cudaEventElapsedTime(&ms,output_start,output_end),"reduce/push elapsed");return ms*1000.;}
};
void run_reduced(Rank& a,Rank& b,Join& join,const std::vector<float>& w,int epoch,Exchange exchange,ExchangeEvents& ev,bool profile=false){
    // Live routes and weights remain inside wall timing; only invariant tables
    // were uploaded during construction. Pinned buffers remain alive until DAG completion.
    a.metadata(epoch,&w);b.metadata(epoch,&w);
    on(0);if(profile) ck(cudaEventRecord(ev.input_start,a.stream),"input push start");
    if(b.count)tp2_input_push(a.x,b.x,a.T*N,a.stream);
    if(profile) ck(cudaEventRecord(ev.input_end,a.stream),"input push end");
    ck(cudaEventRecord(ev.input_ready,a.stream),"input ready");
    a.launch(false,profile);a.reduce(a.local_sum,false);
    on(1);ck(cudaStreamWaitEvent(b.stream,ev.input_ready,0),"peer waits input");
    b.launch(false,profile);
    if(profile) ck(cudaEventRecord(ev.output_start,b.stream),"reduce/push start");
    b.reduce(exchange==Exchange::Reduced?join.peer:b.local_sum,exchange==Exchange::Reduced);
    if(profile) ck(cudaEventRecord(ev.output_end,b.stream),"reduce/push end");
    if(exchange==Exchange::Reduced)ck(cudaEventRecord(ev.peer_ready,b.stream),"peer output ready");
    if(exchange==Exchange::ReducedHost){
        // Attribution control: reduced payload, but original host join/runtime return copy.
        b.sync();on(0);
        if(profile) ck(cudaEventRecord(join.transfer_start,a.stream),"runtime return start");
        ck(cudaMemcpyPeerAsync(join.peer,0,b.local_sum,1,(size_t)a.T*N*sizeof(float),a.stream),"reduced runtime return");
        if(profile) ck(cudaEventRecord(join.transfer_end,a.stream),"runtime return end");
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
// Four dependency events, no host barrier between GU, all-gather and down.
// Each rank pushes its own half into both full-F buffers, records readiness,
// then waits for the other half. Final primary completion also covers the peer,
// making all source/scratch/metadata buffers safe to reuse on the next replay.
void run_row_tp(Rank& a,Rank& b,Join& join,const std::vector<float>& w,int epoch,
                ExchangeEvents& ev,bool profile=false) {
    a.metadata(epoch,&w);b.metadata(epoch,&w);
    on(0);tp2_input_push(a.x,b.x,a.T*N,a.stream);
    ck(cudaEventRecord(ev.input_ready,a.stream),"row input ready");
    a.launch(false,profile);
    tp2_hidden_push(a.hidden_ptr(),a.hidden_ptr(true),b.hidden_ptr(true),a.T*a.K,(F/2/32)*36,0,a.stream);
    ck(cudaEventRecord(ev.hidden_ready0,a.stream),"low hidden ready");
    on(1);ck(cudaStreamWaitEvent(b.stream,ev.input_ready,0),"row peer waits input");
    b.launch(false,profile);
    tp2_hidden_push(b.hidden_ptr(),b.hidden_ptr(true),a.hidden_ptr(true),b.T*b.K,(F/2/32)*36,1,b.stream);
    ck(cudaEventRecord(ev.hidden_ready1,b.stream),"high hidden ready");
    ck(cudaStreamWaitEvent(b.stream,ev.hidden_ready0,0),"peer waits low hidden");
    b.launch_down();b.reduce(join.peer,true);
    ck(cudaEventRecord(ev.peer_ready,b.stream),"row peer reduced ready");
    on(0);ck(cudaStreamWaitEvent(a.stream,ev.hidden_ready1,0),"primary waits high hidden");
    a.launch_down();a.reduce(a.local_sum,false);
    ck(cudaStreamWaitEvent(a.stream,ev.peer_ready,0),"primary waits high output");
    tp2_concat_rows(a.local_sum,join.peer,join.out,a.T,N/2,a.stream);
    a.sync();
}
Results row_result(Rank& a,Rank& b,Join& join) {
    on(0);auto low=download(a.out,(size_t)a.T*a.K*N/2);
    auto combined=download(join.out,(size_t)a.T*N);
    on(1);auto high=download(b.out,(size_t)a.T*a.K*N/2);
    std::vector<float> rows((size_t)a.T*a.K*N);
    for(int i=0;i<a.T*a.K;++i){
        std::copy_n(low.data()+(size_t)i*N/2,N/2,rows.data()+(size_t)i*N);
        std::copy_n(high.data()+(size_t)i*N/2,N/2,rows.data()+(size_t)i*N+N/2);
    }
    return {std::move(rows),std::move(combined),{}};
}
bool run_case(const Fixture& f,int T,const Options& o) {
    const int K=o.experts,epochs=4;
    const bool reduced=o.exchange!=Exchange::Rows;
    const char* exchange=o.exchange==Exchange::Rows?"rows":o.exchange==Exchange::ReducedHost?"reduced-host":"reduced";
    std::printf("CASE GU=%s(%d) DOWN=%s(%d) N=%d F=%d T=%d K=%d mode=%d graph=%d exchange=%s tp_layout=%s order=%s\n",
        strata::ggml_type_name(f.gu),f.gu,strata::ggml_type_name(f.down),f.down,N,F,T,K,o.mode,
        (int)o.graphs,exchange,o.tp_layout.c_str(),o.order.c_str());
    Rank oracle(f,0,T,K,0,K,false,o.graphs,reduced,false,o.mode);Join oracle_join(T,K);
    std::array<Results,4> reference;
    auto oracle_run=[&](int epoch,const std::vector<float>& w,bool profile=false){
        oracle.metadata(epoch,reduced?&w:nullptr);oracle.launch(false,profile);
        if(reduced){oracle.reduce(oracle_join.out,false);oracle.sync();}
        else oracle_join.run(oracle,nullptr,w,false,profile);
    };
    auto oracle_check=[&](int epoch,bool save){
        auto x=input(T,epoch);auto w=routing(T,K,epoch);
        on(0);upload(oracle.x,x,oracle.stream);oracle.poison();oracle.sync();
        oracle_run(epoch,w);
        // Independent original ordered per-expert combine is always the oracle.
        on(0);const auto reduced_output=download(oracle_join.out,(size_t)T*N);
        oracle_join.run(oracle,nullptr,w,false);
        auto result=oracle_join.result();
        if(!compare(result.combined,reduced_output,"single_reduce_vs_old_join",true,false,epoch==2))return false;
        if(save){reference[epoch]=std::move(result);reference[epoch].hq=oracle.hidden();return true;}
        return compare(reference[epoch].rows,result.rows,"oracle_replay_rows",true,false) &&
               compare(reference[epoch].combined,result.combined,"oracle_replay_combine",true,false,epoch==2);
    };
    for(int epoch=0;epoch<epochs;++epoch)if(!oracle_check(epoch,true))return false;
    std::vector<std::pair<std::string,int>> modes={{"EP-balanced",K/2},{"EP-primary-heavy",(3*K+2)/4},
        {"EP-peer-heavy",K-(3*K+2)/4},{"EP-primary-only",K},{"EP-peer-only",0}};
    if(o.tp_layout!="row")modes.emplace_back("TP2-column",-1);
    if(o.tp_layout!="column")modes.emplace_back("TP2-output-row",-2);
    for(const auto& [name,g0]:modes){
        const bool tensor=g0<0,row_tp=g0==-2;
        Rank a(f,0,T,K,0,tensor?K:g0,tensor,o.graphs,reduced,row_tp,o.mode);
        Rank b(f,1,T,K,tensor?0:g0,tensor?K:K-g0,tensor,o.graphs,reduced,row_tp,o.mode);
        Join join(T,K);ExchangeEvents ev;
        std::printf(" MODE %s computed_experts=%d/%d GU_width=%d/%d down_input=%d down_output=%d fixture_weight_bytes=%zu/%zu dynamic_record_bytes=%zu/%zu\n",
            name.c_str(),a.count,b.count,a.ff,b.ff,row_tp?F:a.ff,a.width,
            a.blobs.size()*a.L.bytes,b.blobs.size()*b.L.bytes,a.record_bytes,b.record_bytes);
        auto candidate_run=[&](int epoch,const std::vector<float>& w,bool profile=false){
            if(row_tp)run_row_tp(a,b,join,w,epoch,ev,profile);
            else if(reduced)run_reduced(a,b,join,w,epoch,o.exchange,ev,profile);
            else{
                b.metadata(epoch);on(1);
                if(b.count)ck(cudaMemcpyPeerAsync(b.x,1,a.x,0,(size_t)T*N*sizeof(float),b.stream),"input P2P");
                b.launch(false,profile);a.metadata(epoch);a.launch(false,profile);
                join.run(a,&b,w,tensor,profile);
            }
        };
        auto candidate_check=[&](int epoch,bool verbose){
            if(verbose)std::printf(" CHECK %s epoch=%d\n",name.c_str(),epoch);
            auto x=input(T,epoch);auto w=routing(T,K,epoch);
            on(0);upload(a.x,x,a.stream);a.poison();b.poison();
            on(1);ck(cudaMemsetAsync(b.x,0xff,(size_t)T*N*sizeof(float),b.stream),"poison peer input");
            if(row_tp){
                ck(cudaMemsetAsync(b.down_scratch,0xff,k::native_expert_scratch_bytes(T*K,F),b.stream),"poison high gather");
                on(0);ck(cudaMemsetAsync(a.down_scratch,0xff,k::native_expert_scratch_bytes(T*K,F),a.stream),"poison low gather");
            }
            b.sync();on(0);ck(cudaMemsetAsync(join.peer,0xff,(size_t)T*K*N*sizeof(float),a.stream),"poison peer return");a.sync();
            candidate_run(epoch,w);
            const auto got=row_tp?row_result(a,b,join):reduced?reduced_result(a,b,join,tensor):join.result();
            bool ok=compare(reference[epoch].rows,got.rows,"expert_rows",!tensor||row_tp,verbose) &&
                compare(reference[epoch].combined,got.combined,"test_FP32_combine",row_tp||(!reduced&&!tensor),verbose,epoch==2);
            if(b.count){on(1);const auto received=download(b.x,x.size());
                if(std::memcmp(received.data(),x.data(),x.size()*sizeof(float)))throw std::runtime_error("peer input content mismatch");}
            if(tensor)ok=hidden_exact(reference[epoch].hq,a.hidden(),b.hidden(),T*K,verbose)&&ok;
            if(row_tp)for(Rank* rank:{&a,&b}){
                on(rank->device);const auto gathered=download(rank->hidden_ptr(true),reference[epoch].hq.size());
                if(gathered!=reference[epoch].hq){std::fprintf(stderr,"full hidden gather mismatch rank=%d epoch=%d\n",rank->device,epoch);ok=false;}
            }
            return ok;
        };
        // Correctness is deliberately separate from timing. Check all four live
        // epochs on both sides of the timed batch; do not download every replay.
        for(int epoch=0;epoch<epochs;++epoch)if(!candidate_check(epoch,true))return false;
        double baseline=0,candidate=0;
        for(int iteration=-epochs;iteration<o.iterations;++iteration){
            const int epoch=(iteration+epochs)%epochs;
            auto x=input(T,epoch);auto w=routing(T,K,epoch);
            auto timed=[&](bool single){
                Rank& rank=single?oracle:a;
                on(0);upload(rank.x,x,rank.stream);rank.sync();
                const auto begin=std::chrono::steady_clock::now();
                if(single)oracle_run(epoch,w);else candidate_run(epoch,w);
                const auto end=std::chrono::steady_clock::now();
                if(iteration>=0)(single?baseline:candidate)+=std::chrono::duration<double,std::micro>(end-begin).count();
            };
            // Swap the order for the same live epoch in successive four-epoch
            // cycles, so route/input differences cannot alias with pair order.
            const int sequence=iteration+epochs;
            const bool reverse=o.order=="reverse" ||
                (o.order=="alternate" && ((sequence + sequence/epochs)%2));
            timed(!reverse);timed(reverse);
        }
        for(int epoch=0;epoch<epochs;++epoch)if(!oracle_check(epoch,false)||!candidate_check(epoch,false))return false;
        std::printf(" CORRECTNESS %s PASS pre/post epochs=4 hidden_local=%s hidden_gather=%s rows=%s combine=%s\n",
            name.c_str(),tensor?"byte-exact":"n/a",row_tp?"byte-exact-both-ranks":"n/a",
            (!tensor||row_tp)?"bit-exact":"tolerance",row_tp||(!reduced&&!tensor)?"bit-exact":"tolerance");
        std::printf(" TIMING %s T=%d pairs=%d single_wall_us=%.3f candidate_wall_us=%.3f speedup=%.5f order=%s profile_events=0 input_bytes=%zu hidden_bytes_each_way=%zu return_bytes=%zu exchange=%s\n",
            name.c_str(),T,o.iterations,baseline/o.iterations,candidate/o.iterations,baseline/candidate,o.order.c_str(),
            b.count?(size_t)T*N*sizeof(float):0,row_tp?(size_t)T*K*(F/2/32)*36:0,
            row_tp?(size_t)T*(N/2)*sizeof(float):reduced?(size_t)T*N*sizeof(float):(size_t)b.count*T*N*sizeof(float),exchange);
        if(o.profile_stages){
            // Extra instrumentation is a separate diagnostic replay, never mixed
            // into the wall-time batch. Row rank events measure GU only.
            auto x=input(T,0);auto w=routing(T,K,0);on(0);upload(a.x,x,a.stream);a.sync();
            candidate_run(0,w,true);
            std::printf(" PROFILE %s separate_replay=1 rank_stage=%s rank0_us=%.3f rank1_us=%.3f",
                name.c_str(),row_tp?"GU-only":"full-FFN",a.compute_us(),b.compute_us());
            if(reduced&&!row_tp)std::printf(" input_us=%.3f reduce_push_us=%.3f",ev.input_us(),ev.output_us());
            if(!row_tp&&o.exchange!=Exchange::Reduced)std::printf(" runtime_return_us=%.3f",join.transfer_us());
            std::printf(" (stages are not summed; excludes uninstrumented stages)\n");
        }
    }
    return true;
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
        std::printf("Timing: paired single/candidate order control; only dependency events and final completion in reduced performance path. Correctness pre/post four epochs; --profile-stages is a separate replay.\n");
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
