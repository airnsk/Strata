// Opt-in real-weight whole-GDN-layer diagnostic. Requires two peer-accessible GPUs.
// The -1 reference shares executor orchestration, NOT an independent Verifier.
// A separate token-serial legacy fused-GDN replay checks its recurrence/commit.
// Fixed numerical gates below are engineering parity gates, not model quality.
#include "strata/core/tp_gdn_layer.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/tp_gdn_weights.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/tp_gdn_exchange.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <limits>
#include <numeric>
#include <thread>
#include <new>
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
    int layer=0,mode=8,warmup=3,trials=12,block_calls=16,block_trials=4; bool benchmark=false,calibrate=false,model_probe=false,profile_flat=false,execution_explicit=false;
    bool warmup_explicit=false,trials_explicit=false,rccl_options_explicit=false;
    tp::TpGdnRcclOptions rccl;
    std::string rccl_scenario="normal";
    int rccl_first=0,rccl_delay_ms=200;
};
Options parse(int argc,char**argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--calibrate"){o.calibrate=true;continue;}
        if(a=="--model-probe"){o.model_probe=true;continue;}
        if(a=="--benchmark"){o.benchmark=true;continue;}
        if(a=="--profile-flat"||a=="--profile-stages"){o.profile_flat=true;continue;}
        if(i+1==argc) throw std::invalid_argument("missing argument for "+a);
        const std::string v=argv[++i];
        if(a=="--gguf") o.shards.push_back(v);
        else if(a=="--pack") o.pack=v;
        else if(a=="--execution"){o.execution=v;o.execution_explicit=true;}
        else if(a=="--rccl-library"){o.rccl.library=v;o.rccl_options_explicit=true;}
        else if(a=="--rccl-scenario"){o.rccl_scenario=v;o.rccl_options_explicit=true;}
        else if(a=="--rccl-timeout-ms"||a=="--rccl-init-timeout-ms"||a=="--rccl-launch-first"||a=="--rccl-delay-ms"){
            size_t used=0;const int n=std::stoi(v,&used);if(used!=v.size())throw std::invalid_argument("invalid RCCL integer");
            o.rccl_options_explicit=true;
            if(a=="--rccl-timeout-ms")o.rccl.timeout_ms=n;
            else if(a=="--rccl-init-timeout-ms")o.rccl.init_timeout_ms=n;
            else if(a=="--rccl-launch-first")o.rccl_first=n;
            else o.rccl_delay_ms=n;
        }
        else if(a=="--layer" || a=="--mode" || a=="--bench-warmup" || a=="--bench-trials" || a=="--bench-block-calls" || a=="--bench-block-trials") {
            size_t used=0; int n=std::stoi(v,&used);
            if(used!=v.size()) throw std::invalid_argument("invalid integer");
            if(a=="--layer")o.layer=n;else if(a=="--mode")o.mode=n;else if(a=="--bench-warmup"){o.warmup=n;o.warmup_explicit=true;}else if(a=="--bench-trials"){o.trials=n;o.trials_explicit=true;}else if(a=="--bench-block-calls")o.block_calls=n;else o.block_trials=n;
        } else throw std::invalid_argument("unknown option "+a);
    }
    if(o.shards.empty() || o.pack.empty() || o.layer<0 || o.layer>=48 || o.layer==1 || o.layer%4==3 ||
       (o.mode!=7 && o.mode!=8) || o.warmup<1 || o.warmup>100 || o.trials<2 || o.trials>1000 || o.block_calls<2 || o.block_calls>64 || o.block_trials<2 || o.block_trials>20 ||
       (o.execution!="runtime"&&o.execution!="consolidated"&&o.execution!="captured"&&o.execution!="flat"&&o.execution!="flat-hc"&&o.execution!="column"&&o.execution!="hybrid"&&o.execution!="hybrid-rccl"&&o.execution!="all"))
        throw std::invalid_argument("usage: tp2_gdn_layer --pack PACK --gguf SHARD [--gguf SHARD ...] [--layer non-PLE-GDN-index] [--mode 7|8] [--execution runtime|consolidated|captured|column|hybrid|hybrid-rccl|flat|flat-hc|all] [--benchmark | --calibrate | --model-probe] [--bench-warmup 3] [--bench-trials 12] [--profile-stages] [--bench-block-calls 16] [--bench-block-trials 4]");
    if(o.execution=="hybrid-rccl"){
#ifndef STRATA_TP2_GDN_RCCL
        throw std::invalid_argument("hybrid-rccl requires STRATA_TP2_GDN_RCCL_BUILD=ON");
#endif
        if(o.rccl.library.empty()||o.rccl.library.front()!='/'||o.calibrate||o.model_probe||o.profile_flat||
           o.rccl.timeout_ms<100||o.rccl.timeout_ms>120000||o.rccl.init_timeout_ms<100||o.rccl.init_timeout_ms>120000||
           (o.rccl_first!=0&&o.rccl_first!=1)||o.rccl_delay_ms<1||o.rccl_delay_ms>10000||
           (o.rccl_scenario!="normal"&&o.rccl_scenario!="delayed-rank"&&o.rccl_scenario!="missing-rank")||
           (o.rccl_scenario!="normal"&&o.benchmark)||
           (o.rccl_scenario=="delayed-rank"&&o.rccl_delay_ms>=o.rccl.timeout_ms))
            throw std::invalid_argument("hybrid-rccl needs --rccl-library ABS_PATH, bounded timeouts and unprofiled normal/fault scope; fault timing is forbidden");
    }else if(o.rccl_options_explicit)throw std::invalid_argument("RCCL options require --execution hybrid-rccl");
    if(o.calibrate){
        if(o.benchmark||o.profile_flat||(o.execution_explicit&&o.execution!="captured"))
            throw std::invalid_argument("--calibrate only supports unprofiled output-row captured; do not combine with --benchmark/--profile-stages/other execution");
        o.execution="captured";o.execution_explicit=true;
    }
    if(o.model_probe){
        if(o.calibrate||o.benchmark||o.profile_flat||(o.execution_explicit&&o.execution!="hybrid"))
            throw std::invalid_argument("--model-probe only supports unprofiled hybrid; do not combine with --calibrate/--benchmark/--profile-stages/other execution");
        o.execution="hybrid";o.execution_explicit=true;
        if(!o.warmup_explicit)o.warmup=1;
        if(!o.trials_explicit)o.trials=4;
        if(o.warmup>16||o.trials>64||o.mode!=8)
            throw std::invalid_argument("--model-probe requires mode=8, warmup=1..16 and trials=2..64");
    }
    return o;
}
void hc_source_metadata(const Options& o) {
    // GgufFile mmaps the source but only find()/shape/type directory entries are
    // inspected here. No tensor_data(), payload copying, or device allocation.
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    for(const auto& path:o.shards)files.push_back(std::make_unique<strata::GgufFile>(path));
    for(const char* role:{"hc_attn_norm.weight","hc_attn_down.weight","hc_attn_up.weight","hc_attn_inject.weight",
                         "hc_ffn_norm.weight","hc_ffn_down.weight","hc_ffn_up.weight","hc_ffn_inject.weight"}){
        std::map<uint32_t,int> counts;int missing=0,ambiguous=0;
        for(int layer=0;layer<48;++layer){
            const std::string name="blk."+std::to_string(layer)+"."+role;
            const strata::TensorInfo* tensor=nullptr;int found=0,owner=-1;
            for(size_t i=0;i<files.size();++i)if(const auto* t=files[i]->find(name)){tensor=t;++found;owner=int(i);}
            if(!found)++missing;else if(found>1)++ambiguous;else ++counts[tensor->type];
            if(layer==o.layer){
                std::printf("HC_GGUF_SELECTED layer=%d role=%s matches=%d",layer,role,found);
                if(found==1){std::printf(" shard-index=%d type=%s type-id=%u shape=",owner,strata::ggml_type_name(tensor->type),tensor->type);
                    for(size_t j=0;j<tensor->shape.size();++j)std::printf("%s%llu",j?"x":"",(unsigned long long)tensor->shape[j]);}
                std::printf("\n");
            }
        }
        for(const auto& item:counts)std::printf("HC_GGUF_COUNTS role=%s type=%s type-id=%u layers=%d of=48\n",role,strata::ggml_type_name(item.first),item.first,item.second);
        std::printf("HC_GGUF_COVERAGE role=%s missing-layers=%d ambiguous-layers=%d of=48\n",role,missing,ambiguous);
    }
    std::printf("HC_SOURCE_NOTE GGUF directory types describe supplied sources; runtime canonical HC uses pack-converted BF16 projections and F32 norm, not inferred GGUF quantized dispatch\n");
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
    void set(const std::vector<T>& v) {if(v.size()!=n)throw std::runtime_error("buffer shape");on(device);ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice),"legacy upload");ck(cudaDeviceSynchronize(),"diagnostic upload completion");}
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

void flat_protocol_preflight(Checks& checks) {
#if defined(STRATA_HIP_GFX906)
    if(!__atomic_always_lock_free(sizeof(uint64_t),nullptr))throw std::runtime_error("flat protocol requires lock-free host uint64 atomics");
    if(!k::tp_gdn_protocol_supported())throw std::runtime_error("flat protocol unavailable on this compiler/backend; no model graph launched");
    struct Lifetime {
        k::TpGdnProtocolControl* host=nullptr;
        k::TpGdnProtocolControl* device[2]{};cudaStream_t streams[2]{};void* inbox[2]{};cudaGraph_t graphs[2]{};cudaGraphExec_t execs[2]{};
        void cancel_drain() noexcept {
            if(host)__atomic_store_n(&host->host_abort.value,uint64_t{1},__ATOMIC_RELEASE);
            for(int d=0;d<2;++d){cudaSetDevice(d);if(streams[d])cudaStreamSynchronize(streams[d]);}
        }
        ~Lifetime(){
            cancel_drain();
            for(int d=0;d<2;++d){cudaSetDevice(d);if(execs[d])cudaGraphExecDestroy(execs[d]);if(graphs[d])cudaGraphDestroy(graphs[d]);if(inbox[d])cudaFree(inbox[d]);if(streams[d])cudaStreamDestroy(streams[d]);}
            if(host){host->~TpGdnProtocolControl();hipHostFree(host);}
        }
    } life;
    on(0);ck(hipHostMalloc(reinterpret_cast<void**>(&life.host),sizeof(*life.host),hipHostMallocMapped|hipHostMallocCoherent|hipHostMallocPortable),"preflight coherent mapped control");
    new(life.host) k::TpGdnProtocolControl{};
    constexpr int HALF=3072;std::array<std::unique_ptr<Buffer<uint32_t>>,2> source,full;
    std::array<std::unique_ptr<Buffer<k::TpGdnProtocolStamp>>,2> stamps;
    struct DrainBeforeBuffers {Lifetime& owner;~DrainBeforeBuffers(){owner.cancel_drain();}} drain_guard{life};
    k::TpGdnExchangePacket packets[2];uint64_t ticks[2]{};
    for(int d=0;d<2;++d){
        on(d);auto error=cudaDeviceEnablePeerAccess(1-d,0);if(error==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(error,"preflight enable P2P");
        ck(cudaStreamCreateWithFlags(&life.streams[d],cudaStreamNonBlocking),"preflight stream");
        ck(hipHostGetDevicePointer(reinterpret_cast<void**>(&life.device[d]),life.host,0),"preflight mapped alias");
        ck(hipExtMallocWithFlags(&life.inbox[d],HALF*sizeof(uint32_t),hipDeviceMallocUncached),"preflight uncached inbox");
        int rate=0;ck(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,d),"preflight wall-clock rate");
        if(rate<=0)throw std::runtime_error("invalid constant wall-clock rate");
        ticks[d]=uint64_t(rate)*100; // 100 ms; rate is kHz
        std::printf("FLAT_PREFLIGHT device=%d wall-clock-kHz=%d timeout-ticks=%llu timeout-us=100000\n",d,rate,(unsigned long long)ticks[d]);
        source[d]=std::make_unique<Buffer<uint32_t>>(HALF,d);full[d]=std::make_unique<Buffer<uint32_t>>(2*HALF,d);
        stamps[d]=std::make_unique<Buffer<k::TpGdnProtocolStamp>>(1,d);
    }
    for(int d=0;d<2;++d){auto& p=packets[d];p.count=1;p.spans[0]={source[d]->p,full[d]->p,1,HALF,HALF};p.peer_inbox=life.inbox[1-d];p.own_inbox=life.inbox[d];p.inbox_bytes=HALF*sizeof(uint32_t);}
    uint64_t epoch=0;
    auto initialize=[&](bool preserve_ready=false){
        // All previous work has drained before any signal slot is reused.
        __atomic_store_n(&life.host->epoch.value,++epoch,__ATOMIC_RELEASE);
        __atomic_store_n(&life.host->host_abort.value,uint64_t{0},__ATOMIC_RELEASE);
        for(int d=0;d<2;++d){__atomic_store_n(&life.host->abort[d].value,uint64_t{0},__ATOMIC_RELEASE);
            if(!preserve_ready)for(auto& slot:life.host->ready[d])__atomic_store_n(&slot.value,uint64_t{0},__ATOMIC_RELEASE);
            std::vector<uint32_t> input(HALF);for(int i=0;i<HALF;++i)input[i]=uint32_t(epoch*10000+d*1000+i+1);
            source[d]->set(input);full[d]->set(std::vector<uint32_t>(2*HALF,0xdeadbeefu));stamps[d]->set({{}});
            on(d);ck(cudaDeviceSynchronize(),"preflight initialization completion");}
    };
    auto launch=[&](int d,int phase){on(d);k::tp_gdn_packet_push(packets[d],life.device[d],d,life.streams[d]);
        k::tp_gdn_protocol_wait(life.device[d],stamps[d]->p,d,phase,ticks[d],life.streams[d]);
        k::tp_gdn_packet_unpack(packets[d],life.device[d],stamps[d]->p,d,phase,life.streams[d]);};
    auto drain=[&](int mask,bool failed){
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);bool host_timeout=false;
        for(;;){bool ready=true;for(int d=0;d<2;++d)if(mask&(1<<d)){on(d);const auto e=cudaStreamQuery(life.streams[d]);if(e==cudaErrorNotReady)ready=false;else ck(e,"preflight stream query");}
            if(ready)break;
            if(std::chrono::steady_clock::now()>=deadline){host_timeout=true;break;}std::this_thread::sleep_for(std::chrono::milliseconds(1));}
        // Publish host cancellation before any blocking drain on a failure.
        if(failed||host_timeout)__atomic_store_n(&life.host->host_abort.value,epoch,__ATOMIC_RELEASE);
        for(int d=0;d<2;++d)if(mask&(1<<d)){on(d);ck(cudaStreamSynchronize(life.streams[d]),"preflight drained stream");}
        if(host_timeout)throw std::runtime_error("flat protocol preflight exceeded independent host deadline");
    };
    auto status=[&](int d,k::TpGdnProtocolStatus expected){const auto s=stamps[d]->get().at(0);
        std::printf("FLAT_PREFLIGHT_STAMP rank=%d epoch=%llu status=%llu polls=%llu elapsed-ticks=%llu\n",d,(unsigned long long)s.epoch,(unsigned long long)s.status,(unsigned long long)s.polls,(unsigned long long)s.elapsed_ticks);
        checks.require(s.epoch==epoch&&s.status==uint64_t(expected),"flat preflight exact epoch/status");};
    for(int phase=0;phase<k::kTpGdnProtocolPhases;++phase){initialize();launch(0,phase);launch(1,phase);drain(3,false);
        std::vector<uint32_t> expected;for(int d=0;d<2;++d){auto v=source[d]->get();expected.insert(expected.end(),v.begin(),v.end());}
        for(int d=0;d<2;++d){status(d,k::TpGdnProtocolStatus::Ready);checks.exact("flat preflight actual peer data",expected,full[d]->get());}}
    // Capture the actual producer/wait/unpack sequence, with multiple producer
    // blocks, mapped Y runs, and the two-span routed/shared hidden packet.
    for(int shape=0;shape<2;++shape){
        const int active=shape?990:HALF;
        for(int d=0;d<2;++d){auto& p=packets[d];p.count=shape?2:1;p.inbox_bytes=size_t(active)*sizeof(uint32_t);
            if(shape){p.spans[0]={source[d]->p,full[d]->p,10,90,90};p.spans[1]={source[d]->p+900,full[d]->p+1800,1,90,90};}
            else{p.spans[0]={source[d]->p,full[d]->p,1,HALF,1024};p.spans[1]={};}
            on(d);if(life.execs[d]){ck(cudaGraphExecDestroy(life.execs[d]),"preflight old executable");life.execs[d]={};}
            if(life.graphs[d]){ck(cudaGraphDestroy(life.graphs[d]),"preflight old graph");life.graphs[d]={};}
            ck(cudaStreamBeginCapture(life.streams[d],cudaStreamCaptureModeGlobal),"preflight begin capture");launch(d,2);
            ck(cudaStreamEndCapture(life.streams[d],&life.graphs[d]),"preflight end capture");
            ck(cudaGraphInstantiate(&life.execs[d],life.graphs[d],nullptr,nullptr,0),"preflight instantiate");}
        for(int replay=0;replay<3;++replay){initialize(replay!=0);
            for(int d=0;d<2;++d){on(d);ck(cudaGraphLaunch(life.execs[d],life.streams[d]),"preflight graph replay");}drain(3,false);
            std::vector<uint32_t> expected(size_t(active)*2);size_t offset=0;
            for(int span=0;span<packets[0].count;++span){const auto& spec=packets[0].spans[span];
                for(int d=0;d<2;++d){const auto src=read(static_cast<const uint32_t*>(packets[d].spans[span].src),size_t(spec.rows)*spec.half_words,d);
                    for(int row=0;row<spec.rows;++row)for(int j=0;j<spec.half_words;++j){const size_t dst=offset+size_t(row)*2*spec.half_words+(j/spec.run_words)*2*spec.run_words+d*spec.run_words+j%spec.run_words;expected[dst]=src[size_t(row)*spec.half_words+j];}}
                offset+=size_t(spec.rows)*2*spec.half_words;}
            std::printf("FLAT_PREFLIGHT_CAPTURE shape=%s replay=%d retained-old-ready=%d source-words-per-rank=%d\n",shape?"two-span-hidden":"mapped-Y",replay,int(replay!=0),active);
            for(int d=0;d<2;++d){status(d,k::TpGdnProtocolStatus::Ready);checks.exact("flat captured peer data",expected,read(full[d]->p,size_t(active)*2,d));}}
    }
    for(int d=0;d<2;++d){auto& p=packets[d];p.count=1;p.spans[0]={source[d]->p,full[d]->p,1,HALF,HALF};p.spans[1]={};p.inbox_bytes=HALF*sizeof(uint32_t);}
    // Acquire-observe the waiter entering this generation before releasing a
    // host cancellation. A completed timeout would NOT validate this edge.
    initialize();launch(0,0);const auto entered_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
    while(__atomic_load_n(&life.host->ready[0][0].value,__ATOMIC_ACQUIRE)!=epoch&&std::chrono::steady_clock::now()<entered_deadline)std::this_thread::yield();
    const bool entered=__atomic_load_n(&life.host->ready[0][0].value,__ATOMIC_ACQUIRE)==epoch;
    __atomic_store_n(&life.host->host_abort.value,epoch,__ATOMIC_RELEASE);drain(1,true);
    checks.require(entered,"host abort targets an entered GPU wait");status(0,k::TpGdnProtocolStatus::Aborted);
    checks.exact("host abort zeros output",std::vector<uint32_t>(2*HALF),full[0]->get());
    initialize();launch(0,0);drain(1,true);status(0,k::TpGdnProtocolStatus::TimedOut);
    checks.exact("flat missing peer zeros output",std::vector<uint32_t>(2*HALF),full[0]->get());
    // A rank arriving only after its peer's timeout must observe sticky abort.
    launch(1,0);drain(2,true);status(1,k::TpGdnProtocolStatus::Aborted);
    checks.exact("flat late peer zeros output",std::vector<uint32_t>(2*HALF),full[1]->get());
    initialize();__atomic_store_n(&life.host->ready[1][0].value,epoch+1,__ATOMIC_RELEASE);launch(0,0);drain(1,true);
    status(0,k::TpGdnProtocolStatus::InvalidEpoch);checks.exact("flat future epoch zeros output",std::vector<uint32_t>(2*HALF),full[0]->get());
    initialize();__atomic_store_n(&life.host->ready[0][0].value,epoch,__ATOMIC_RELEASE);launch(0,0);drain(1,true);
    status(0,k::TpGdnProtocolStatus::InvalidEpoch);checks.exact("flat duplicate epoch zeros output",std::vector<uint32_t>(2*HALF),full[0]->get());
    if(checks.failures)throw std::runtime_error("flat model-free preflight failed; no model graph launched");
    std::printf("FLAT_PREFLIGHT PASS peer-data/all-phase/captured-mapped/two-span/reused-generation/inflight-host-abort/missing/late/future/duplicate-epoch; streams drained before model loading\n");
#else
    (void)checks;throw std::runtime_error("flat protocol preflight requires the supported HIP runtime");
#endif
}
std::vector<float> signed_input(size_t n,int salt,float scale) {
    // Nonzero deterministic signed values, head/channel/tap-varying. Integer
    // construction avoids implementation-defined random distributions.
    std::vector<float> v(n);
    for(size_t i=0;i<n;++i){const int z=int((i*73+size_t(salt)*131+(i/128)*29)%2003)-1001;v[i]=scale*float(z==0?1:z)/1001.f;}
    return v;
}
void same_input_ffn_oracle(Checks&,const tp::TpGdnRankWeights&,const tp::TpGdnLayerSnapshot&);
bool same_float_bits(const std::vector<float>& a,const std::vector<float>& b) {
    return !a.empty()&&a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
}
void column_local_fusion_gate(Checks& c,const tp::TpGdnLayerSnapshot& s) {
    if(s.local_output_partial.empty())return;
    if(s.local_output_partial.size()!=size_t(s.tokens)*N||s.local_shared_partial.size()!=s.local_output_partial.size()||
       s.local_expert_parts.size()!=size_t(s.tokens)*K*N||s.shared_gate_logits.size()!=size_t(s.tokens)||s.route_weights.size()!=size_t(s.tokens)*K)
        throw std::runtime_error("column local diagnostic shape mismatch");
    std::vector<float> expected(s.local_output_partial.size());
    for(int t=0;t<s.tokens;++t)for(int j=0;j<N;++j){const size_t pos=size_t(t)*N+j;
        double value=double(s.local_shared_partial[pos])/(1.0+std::exp(-double(s.shared_gate_logits[t])));
        for(int e=0;e<K;++e)value+=double(s.route_weights[size_t(t)*K+e])*s.local_expert_parts[(size_t(t)*K+e)*N+j];
        expected[pos]=float(value);}
    c.floats("column fused vs local unfused",expected,s.local_output_partial,ffn_gate);
}
void seams(Checks& c,const tp::TpGdnLayerSnapshot& a,const tp::TpGdnLayerSnapshot& b,bool replica=false,bool same_epoch=true,bool association_change=false,const tp::TpGdnRankWeights* oracle=nullptr,bool hybrid=false) {
    if(hybrid)c.require(!b.local_output_partial.empty(),"hybrid local fused diagnostic present");
    if(replica&&!a.local_output_partial.empty())c.require(!b.local_output_partial.empty(),"replica local fused diagnostic present");
    column_local_fusion_gate(c,a);column_local_fusion_gate(c,b);
    c.require(a.tokens==b.tokens&&(!same_epoch||a.epoch==b.epoch),same_epoch?"snapshot token/epoch identity":"snapshot token identity (independent epochs)");
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
    const bool same_input=same_float_bits(a.ffn_input,b.ffn_input);
    if(hybrid)c.exact("hybrid original FFN input",a.ffn_input,b.ffn_input);
    c.exact("ordered router IDs",a.ids,b.ids);
    if(association_change&&!hybrid&&!same_input){
        // PREDECLARED: reuse hc_gate {2e-6,2e-5,5e-6}; no fitted thresholds.
        c.floats("router logits changed input",a.route_logits,b.route_logits,hc_gate);
        c.floats("router weights changed input",a.route_weights,b.route_weights,hc_gate);
        auto diagnostic=[&](const char* label,const std::vector<uint8_t>& x,const std::vector<uint8_t>& y){
            c.require(!x.empty()&&x.size()==y.size(),"original-reference hidden Q8 shape");size_t different=0;
            if(x.size()==y.size())for(size_t i=0;i<x.size();++i)different+=x[i]!=y[i];
            std::printf("DIAGNOSTIC %s original-input-vs-candidate-input byte-differences=%zu/%zu; exact ownership checked by candidate-same-input full-GU oracle\n",label,different,x.size());
        };
        diagnostic("routed Q8",a.routed_hidden_q8,b.routed_hidden_q8);diagnostic("shared Q8",a.shared_hidden_q8,b.shared_hidden_q8);
    }else{
        c.exact("router logits",a.route_logits,b.route_logits);c.exact("router weights",a.route_weights,b.route_weights);
        c.exact("routed hidden Q8 bytes",a.routed_hidden_q8,b.routed_hidden_q8);c.exact("shared hidden Q8 bytes",a.shared_hidden_q8,b.shared_hidden_q8);
    }
    if(association_change||hybrid){
        if(!oracle)throw std::logic_error("column/hybrid parity requires candidate-same-input full-FFN oracle");
        same_input_ffn_oracle(c,*oracle,b);
    }
    c.floats("committed recurrence",a.state,b.state,replica?Gate{0,0,0}:gdn_gate);
    c.floats("committed conv",a.conv,b.conv,replica?Gate{0,0,0}:gdn_gate);
}
void exact_hybrid(Checks& c,const tp::TpGdnLayerSnapshot& a,const tp::TpGdnLayerSnapshot& b,bool same_epoch=true) {
    c.require(a.tokens==b.tokens&&(!same_epoch||a.epoch==b.epoch),"RCCL exact snapshot identity");
    // Every materialized seam and rank-local diagnostic must retain its bits.
    // This is stricter than the unchanged original-full numerical contract.
#define RCCL_EXACT(field) c.exact("RCCL exact " #field,a.field,b.field)
    RCCL_EXACT(residual);RCCL_EXACT(attention_input);RCCL_EXACT(gdn_output);
    RCCL_EXACT(mixer_output);RCCL_EXACT(ffn_input);RCCL_EXACT(route_weights);
    RCCL_EXACT(route_logits);RCCL_EXACT(output);RCCL_EXACT(state);RCCL_EXACT(conv);
    RCCL_EXACT(shared_output);RCCL_EXACT(shared_gate_logits);RCCL_EXACT(expert_parts);
    RCCL_EXACT(local_output_partial);RCCL_EXACT(local_shared_partial);RCCL_EXACT(local_expert_parts);
    RCCL_EXACT(ids);RCCL_EXACT(routed_hidden_q8);RCCL_EXACT(shared_hidden_q8);
#undef RCCL_EXACT
}
void exact_hybrid_layers(Checks& c,const tp::TpGdnLayer& old,const tp::TpGdnLayer& candidate) {
    for(int rank=0;rank<2;++rank){std::printf("RCCL_EXACT_OWNER rank=%d\n",rank);exact_hybrid(c,old.snapshot(rank),candidate.snapshot(rank));}
    if(c.failures)throw std::runtime_error("RCCL transport changed old-hybrid bits; timing prohibited");
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
void matrix_columns(Checks& c,const char* label,const tp::TpNativeMatrix& full,const tp::TpNativeMatrix& shard,
                    const std::vector<int>& columns,int full_device,int shard_device) {
    int block=0,bytes=0;
    if(!strata::block_geometry(full.type,block,bytes)||block<=0||full.input%block||columns.size()%size_t(block))
        throw std::runtime_error("column coherence block geometry");
    const bool shape_ok=full.type==shard.type&&shard.input==int(columns.size())&&full.output==shard.output;
    c.require(shape_ok,label);if(!shape_ok)throw std::runtime_error("column coherence descriptor mismatch");
    const size_t source_pitch=size_t(full.input/block)*bytes,destination_pitch=columns.size()/block*bytes;
    auto source=read(static_cast<const uint8_t*>(full.data),size_t(full.output)*source_pitch,full_device);
    auto actual=read(static_cast<const uint8_t*>(shard.data),size_t(shard.output)*destination_pitch,shard_device);
    std::vector<uint8_t> expected(actual.size());
    for(size_t b=0;b<columns.size();b+=block){
        if(columns[b]<0||columns[b]>=full.input||columns[b]%block)throw std::runtime_error("column coherence split block");
        for(int j=0;j<block;++j)if(columns[b+j]!=columns[b]+j)throw std::runtime_error("column coherence reordered inside block");
        for(int row=0;row<full.output;++row)std::memcpy(expected.data()+size_t(row)*destination_pitch+(b/block)*bytes,
            source.data()+size_t(row)*source_pitch+size_t(columns[b]/block)*bytes,bytes);
    }
    c.exact(label,expected,actual);
}
void expert_byte_coherence(Checks& c,const tp::TpGdnRankWeights& full,const tp::TpGdnRankWeights& shard) {
    const bool columns=shard.partition==tp::TpGdnPartition::InputColumns||shard.partition==tp::TpGdnPartition::HybridRowsColumns;
    const auto& fl=full.gu_layout;const auto& sl=shard.gu_layout;const auto& dl=shard.down_layout;
    const size_t gu_bytes=size_t(F/2)*fl.gu_row,down_bytes=size_t(columns?N:N/2)*dl.d_row;
    c.require(fl.gu_type==sl.gu_type&&fl.d_type==dl.d_type&&sl.gu_row==fl.gu_row&&sl.up_off==gu_bytes&&dl.down_off==2*gu_bytes&&
        shard.expert_bytes==2*gu_bytes+down_bytes&&dl.n_embd==(columns?N:N/2)&&dl.n_ff==(columns?F/2:F),"expert actual raw shard descriptor");
    if(fl.down_off+size_t(N)*fl.d_row>full.expert_bytes||fl.up_off+size_t(F)*fl.gu_row>full.expert_bytes||
       shard.expert_bytes!=2*gu_bytes+down_bytes||sl.up_off!=gu_bytes||dl.down_off!=2*gu_bytes||
       fl.gu_type!=sl.gu_type||fl.d_type!=dl.d_type||sl.gu_row!=fl.gu_row||
       dl.n_embd!=(columns?N:N/2)||dl.n_ff!=(columns?F/2:F))
        throw std::runtime_error("expert coherence cannot safely inspect inconsistent descriptor");
    size_t bad_experts=0,different_bytes=0;
    for(int expert=0;expert<512;++expert){
        const auto source=read(full.expert_arena+size_t(expert)*full.expert_bytes,full.expert_bytes,full.device);
        const auto actual=read(shard.expert_arena+size_t(expert)*shard.expert_bytes,shard.expert_bytes,shard.device);
        std::vector<uint8_t> expected(shard.expert_bytes);
        std::memcpy(expected.data(),source.data()+size_t(shard.rank)*gu_bytes,gu_bytes);
        std::memcpy(expected.data()+sl.up_off,source.data()+fl.up_off+size_t(shard.rank)*gu_bytes,gu_bytes);
        if(columns){
            if(dl.d_row*2!=fl.d_row)throw std::runtime_error("expert coherence column row pitch");
            for(int row=0;row<N;++row)std::memcpy(expected.data()+dl.down_off+size_t(row)*dl.d_row,
                source.data()+fl.down_off+size_t(row)*fl.d_row+size_t(shard.rank)*dl.d_row,dl.d_row);
        }else std::memcpy(expected.data()+dl.down_off,source.data()+fl.down_off+size_t(shard.rank)*down_bytes,down_bytes);
        size_t mismatches=0;for(size_t i=0;i<actual.size();++i)mismatches+=expected[i]!=actual[i];
        bad_experts+=mismatches!=0;different_bytes+=mismatches;
    }
    std::printf("EXPERT_BYTE_COHERENCE partition=%s rank=%d experts=512 bad-experts=%zu differing-bytes=%zu\n",shard.partition==tp::TpGdnPartition::HybridRowsColumns?"hybrid-rows-columns":(columns?"input-columns":"output-rows"),shard.rank,bad_experts,different_bytes);
    c.require(!bad_experts,"all512 actual expert shard bytes");
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
    if(r.partition==tp::TpGdnPartition::InputColumns)matrix_columns(c,"actual mapped output columns",full.out,r.out,value,full.device,r.device);
    else matrix("actual output rows",full.out,r.out,sequence(r.rank*N/2,N/2));
    matrix("actual shared gate rows",full.shared_gate,r.shared_gate,sequence(r.rank*F/2,F/2));
    matrix("actual shared up rows",full.shared_up,r.shared_up,sequence(r.rank*F/2,F/2));
    if(r.partition==tp::TpGdnPartition::InputColumns||r.partition==tp::TpGdnPartition::HybridRowsColumns)matrix_columns(c,"actual shared down columns",full.shared_down,r.shared_down,sequence(r.rank*F/2,F/2),full.device,r.device);
    else matrix("actual shared down rows",full.shared_down,r.shared_down,sequence(r.rank*N/2,N/2));
    mapped_bytes(c,"actual alpha head rows",full.alpha,r.alpha,N*2,heads,full.device,r.device);
    mapped_bytes(c,"actual beta head rows",full.beta,r.beta,N*2,heads,full.device,r.device);
    mapped_bytes(c,"actual dt head values",full.dt,r.dt,4,heads,full.device,r.device);
    mapped_bytes(c,"actual A head values",full.a,r.a,4,heads,full.device,r.device);
    mapped_bytes(c,"actual convolution rows",full.conv,r.conv,4*4,qkv,full.device,r.device);
    c.exact("actual replicated norm",read(full.norm,S,full.device),read(r.norm,S,r.device));
    c.require(g.ssm_k_heads==HK&&g.ssm_v_heads==HV,"expected actual head geometry");
    expert_byte_coherence(c,full,r);
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
std::vector<float> legacy_shared_and_combine(Checks& c,const tp::TpGdnRankWeights& w,const tp::TpGdnLayerSnapshot& s) {
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
    c.exact("same-input shared hidden Q8",s.shared_hidden_q8,read(q.p,size_t(s.tokens)*(F/32)*36,w.device));
    c.floats("CPU FFN combine formula",combined,s.output,ffn_gate);
    return shared;
}
std::vector<float> legacy_routed(Checks& c,const tp::TpGdnRankWeights& w,const tp::TpGdnLayerSnapshot& s) {
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
    c.exact("same-input routed hidden Q8",ordered,s.routed_hidden_q8);
    return parts;
}
void same_input_ffn_oracle(Checks& c,const tp::TpGdnRankWeights& w,const tp::TpGdnLayerSnapshot& candidate) {
    // This independently evaluates ORIGINAL full GU/down/shared weights using
    // the candidate activation and route weights. The fused candidate FFN sum
    // is checked directly, never inferred from diagnostic recomputed parts.
    const auto shared=legacy_shared_and_combine(c,w,candidate),parts=legacy_routed(c,w,candidate);
    std::vector<float> expected(shared.size());
    for(int t=0;t<candidate.tokens;++t)for(int j=0;j<N;++j){double v=shared[size_t(t)*N+j];
        for(int e=0;e<K;++e)v+=double(candidate.route_weights[size_t(t)*K+e])*parts[(size_t(t)*K+e)*N+j];
        expected[size_t(t)*N+j]=float(v);}
    c.floats("same-input original full FFN",expected,candidate.output,ffn_gate);
}
void committed(Checks& c,const tp::TpGdnLayerSnapshot& a,const tp::TpGdnLayerSnapshot& b,const LegacyGdnResult& legacy) {
    c.floats("commit full/TP recurrence",a.state,b.state,gdn_gate);
    c.floats("commit full/TP conv",a.conv,b.conv,gdn_gate);
    c.floats("commit legacy recurrence",legacy.state,b.state,gdn_gate);
    c.floats("commit legacy conv",legacy.conv,b.conv,gdn_gate);
}

const char* execution_name(tp::TpGdnExecution mode) {
    switch(mode){case tp::TpGdnExecution::RuntimeCopies:return "runtime";case tp::TpGdnExecution::Consolidated:return "consolidated";case tp::TpGdnExecution::Captured:return "captured";case tp::TpGdnExecution::FlatCaptured:return "flat";case tp::TpGdnExecution::FlatHcCaptured:return "flat-hc";case tp::TpGdnExecution::ColumnCaptured:return "column";case tp::TpGdnExecution::HybridCaptured:return "hybrid";case tp::TpGdnExecution::HybridRcclCaptured:return "hybrid-rccl";}
    throw std::invalid_argument("unknown execution mode");
}
std::vector<tp::TpGdnExecution> execution_modes(const Options& o) {
    if(o.benchmark&&!o.execution_explicit)return {tp::TpGdnExecution::Captured,tp::TpGdnExecution::ColumnCaptured};
    if(o.execution=="all")return {tp::TpGdnExecution::RuntimeCopies,tp::TpGdnExecution::Consolidated,tp::TpGdnExecution::Captured,tp::TpGdnExecution::FlatCaptured,tp::TpGdnExecution::FlatHcCaptured,tp::TpGdnExecution::ColumnCaptured};
    if(o.execution=="hybrid-rccl")return {tp::TpGdnExecution::HybridCaptured,tp::TpGdnExecution::HybridRcclCaptured};
    if(o.execution=="hybrid")return o.benchmark?std::vector<tp::TpGdnExecution>{tp::TpGdnExecution::Captured,tp::TpGdnExecution::HybridCaptured}:std::vector<tp::TpGdnExecution>{tp::TpGdnExecution::HybridCaptured};
    if(o.execution=="column")return {tp::TpGdnExecution::ColumnCaptured};
    if(o.execution=="flat-hc")return {tp::TpGdnExecution::FlatHcCaptured};
    if(o.execution=="flat")return {tp::TpGdnExecution::FlatCaptured};
    if(o.execution=="captured")return {tp::TpGdnExecution::Captured};
    if(o.execution=="consolidated")return {tp::TpGdnExecution::Consolidated};
    return {tp::TpGdnExecution::RuntimeCopies};
}


void execution_metadata(tp::TpGdnExecution mode) {
    const bool flat=mode==tp::TpGdnExecution::FlatCaptured||mode==tp::TpGdnExecution::FlatHcCaptured;
    const bool column=mode==tp::TpGdnExecution::ColumnCaptured;
    const bool hybrid=mode==tp::TpGdnExecution::HybridCaptured||mode==tp::TpGdnExecution::HybridRcclCaptured;
    const bool graph=flat||column||hybrid||mode==tp::TpGdnExecution::Captured;
    std::printf("EXECUTION_LAYOUT mode=%s full-proposal-graph-launches=%d full-commit-graph-launches=%d tp-proposal-graph-launches-per-rank=%d tp-commit-graph-launches-per-rank=%d positive-keep-only; graph-counts-exclude-copy/host-commands\n",execution_name(mode),graph?1:0,graph?1:0,(flat||mode==tp::TpGdnExecution::HybridRcclCaptured)?1:(column?3:(hybrid?4:(graph?5:0))),graph?1:0);
    if(mode==tp::TpGdnExecution::HybridRcclCaptured)std::printf("RCCL_LAYER_TRANSPORT one-complete-proposal-graph-per-rank; Y=raw-mapped-gather output=raw-row-gather FFN=rank0-then-rank1-fadd_rn; no-host-phase-joins; end-of-proposal-plan-status-only; worker-dispatch-and-completion-in-wall; replicated-original-HC; no-new-quantization\n");
    if(mode==tp::TpGdnExecution::HybridCaptured)std::printf("HYBRID_TRANSPORT row-attention Y/projection gathers; local-hidden320 column-FFN fused down; three joins/four segments vs row four/five; one fewer join, wall-time saving unmeasured; no new GPU kernel; expert-parts snapshot diagnostic recomputation\n");
    if(column)std::printf("COLUMN_TRANSPORT local-Y3072/local-hidden320; fullN partial outputs; two reductions; no Y/hidden gather in timed execution; expert parts snapshot is diagnostic recomputation\n");
    std::printf("HC_OWNERSHIP mode=%s TP=%s full-reference=original-full-HC norm/injection=replicated weight-allocation=full-canonical-BF16-on-each-rank\n",execution_name(mode),mode==tp::TpGdnExecution::FlatHcCaptured?"down160-rows-per-rank/up1280-channels-in-each-of-four-streams-per-rank":"replicated-full-compute");
}
void profile_report(Checks& checks,const tp::TpGdnLayer& layer) {
    std::printf("PROFILE_SCOPE mode=%s separate instrumented graph; rank-local stamps only; producer-to-consumer intervals include host skew/event acquisition, not isolated transport\n",execution_name(layer.execution()));
    for(int rank=0;rank<layer.ranks();++rank){const auto p=layer.profile(rank);
        checks.require(p.labels.size()==p.nanoseconds.size(),"profile label/stamp shape");
        uint64_t previous=0;std::vector<size_t> active;
        for(size_t i=0;i<std::min(p.labels.size(),p.nanoseconds.size());++i)if(p.nanoseconds[i])active.push_back(i);
        std::stable_sort(active.begin(),active.end(),[&](size_t a,size_t b){return p.nanoseconds[a]<p.nanoseconds[b];});
        for(size_t i:active){
            std::printf("CAPTURE_PROFILE mode=%s rank=%d T=%d epoch=%llu phase=%s gpu-ns=%llu since-previous-active-ns=%llu diagnostic-instrumented-only\n",execution_name(layer.execution()),rank,p.tokens,(unsigned long long)p.epoch,p.labels[i].c_str(),(unsigned long long)p.nanoseconds[i],(unsigned long long)(previous&&p.nanoseconds[i]>=previous?p.nanoseconds[i]-previous:0));previous=p.nanoseconds[i];}
        for(size_t i=0;i<p.wait_status.size();++i)if(p.wait_status[i])std::printf("CAPTURE_PROFILE_WAIT mode=%s rank=%d phase=%zu status=%llu wait-ns=%llu polls=%llu\n",execution_name(layer.execution()),rank,i,(unsigned long long)p.wait_status[i],(unsigned long long)p.wait_nanoseconds[i],(unsigned long long)p.wait_polls[i]);
    }
}
void flat_failure_gate(Checks& c,const Options& o,const strata::core::ModelGeometry& g,
                       const tp::TpGdnRankWeights& rank0,const tp::TpGdnRankWeights& rank1,
                       const std::vector<float>& state,const std::vector<float>& conv) {
    for(auto mode:execution_modes(o))if(mode==tp::TpGdnExecution::FlatCaptured||mode==tp::TpGdnExecution::FlatHcCaptured){
        for(bool late:{false,true})for(int rank=0;rank<2;++rank){
            // Each failed session is destroyed, never reset/reused. Preparation
            // and fault setup happen before the bounded proposal is attempted.
            tp::TpGdnLayer fault(g,rank0,&rank1,1,o.mode);
            fault.prepare_flat(1,mode==tp::TpGdnExecution::FlatHcCaptured,false,100000);
            fault.set_execution(mode);fault.reset_state(state,conv);
            fault.set_flat_fault_for_test(late?-1:rank,late?rank:-1,late?300000:0);
            bool rejected=false;std::string reason;const auto start=std::chrono::steady_clock::now();
            try{fault.propose(signed_input(H*N,1901,0.65f),1,1);}catch(const std::exception& e){rejected=true;reason=e.what();}
            const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            std::printf("FLAT_LAYER_FAULT mode=%s kind=%s rank=%d timeout-us=100000 delay-us=%d rejected=%d wall-ms=%.6f reason=%s\n",execution_name(mode),late?"late":"missing",rank,late?300000:0,int(rejected),wall,reason.c_str());
            c.require(rejected,"faulted proposal rejected");
            c.require(reason.find("flat protocol aborted/timed out")!=std::string::npos,"fault rejected for protocol timeout/abort");
            c.require(wall<=5000.0,"faulted proposal drains within 5s host budget");
            auto rejects=[&](const char* label,auto action){bool blocked=false;try{action();}catch(const std::exception&){blocked=true;}c.require(blocked,label);};
            rejects("faulted snapshot unavailable",[&](){(void)fault.snapshot();});
            rejects("faulted residual unavailable",[&](){(void)fault.residual();});
            rejects("faulted commit unavailable",[&](){fault.commit(1);});
            rejects("faulted zero-commit unavailable",[&](){fault.commit(0);});
            rejects("faulted checkpoint reset unavailable",[&](){fault.reset_state(state,conv);});
            rejects("faulted next proposal unavailable",[&](){fault.propose(signed_input(H*N,1902,0.6f),1,2);});
            // The executor must publish cancellation and drain before throwing.
            // A clean device fence after return verifies no accepted work is
            // left pending; it is outside all benchmark clocks.
            for(int d=0;d<2;++d){on(d);ck(cudaDeviceSynchronize(),"fault gate clean device drain");}
            if(c.failures)throw std::runtime_error("flat fail-closed full-layer gate failed; benchmarks prohibited");
        }
    }
    std::printf("FLAT_LAYER_FAULT_GATE PASS missing/late both ranks, no output/commit/reset/reuse accepted, clean drain; following correctness uses fresh sessions\n");
}
void rccl_lifecycle_gate(Checks& c,const Options& o,const strata::core::ModelGeometry& g,
                         const tp::TpGdnRankWeights& rank0,const tp::TpGdnRankWeights& rank1,
                         const std::vector<float>& state,const std::vector<float>& conv) {
    for(int life=0;life<2;++life){
        {
            tp::TpGdnLayer old(g,rank0,&rank1,8,o.mode),candidate(g,rank0,&rank1,8,o.mode);
            for(int t:{1,8}){old.prepare_captured(t);candidate.prepare_rccl(t,o.rccl);}
            old.set_execution(tp::TpGdnExecution::HybridCaptured);
            candidate.set_execution(tp::TpGdnExecution::HybridRcclCaptured);
            uint64_t epoch=0;
            for(int first:{0,1})for(int t:{1,8}){
                old.reset_state(state,conv);candidate.reset_state(state,conv);
                candidate.set_rccl_launch_for_test(first);
                // Two evolving calls exercise both captured physical banks on
                // every T/order, without reset between them. T changes reuse the
                // same capacity buffers and communicators.
                for(int step=0;step<2;++step){
                    const auto input=signed_input(size_t(t)*H*N,7001+life*101+first*31+t*7+step,0.65f);
                    old.propose(input,t,++epoch);candidate.propose(input,t,epoch);
                    exact_hybrid_layers(c,old,candidate);
                    old.commit(t);candidate.commit(t);exact_hybrid_layers(c,old,candidate);
                }
            }
        } // A lifecycle passes only after graph/communicator/buffer destruction.
        std::printf("RCCL_LAYER_LIFECYCLE_PASS life=%d orders=2 tokens=1,8 banks=2 graph_before_comm=1 comm_before_buffers=1\n",life);
    }
}
void rccl_fault_gate(const Options& o) {
    strata::core::ModelGeometry g;tp::TpGdnWeights weights0,weights1;std::string error;
    if(!weights0.load(o.shards,o.pack,g,o.layer,0,0,error,tp::TpGdnPartition::HybridRowsColumns))throw std::runtime_error(error);
    if(!weights1.load(o.shards,o.pack,g,o.layer,1,1,error,tp::TpGdnPartition::HybridRowsColumns))throw std::runtime_error(error);
    Checks c;
    {
        tp::TpGdnLayer old(g,weights0.weights(),&weights1.weights(),1,o.mode),candidate(g,weights0.weights(),&weights1.weights(),1,o.mode);
        old.prepare_captured(1);candidate.prepare_rccl(1,o.rccl);
        old.set_execution(tp::TpGdnExecution::HybridCaptured);candidate.set_execution(tp::TpGdnExecution::HybridRcclCaptured);
        const auto state=signed_input(STATE,17,0.015f),conv=signed_input(CONV,31,0.12f);
        old.reset_state(state,conv);candidate.reset_state(state,conv);
        const int other=1-o.rccl_first;const bool missing=o.rccl_scenario=="missing-rank";
        candidate.set_rccl_launch_for_test(o.rccl_first,missing?other:-1,missing?-1:other,uint64_t(o.rccl_delay_ms)*1000);
        const auto input=signed_input(H*N,7901,0.65f);
        old.propose(input,1,1);
        std::printf("RCCL_LAYER_FAULT_ARMED scenario=%s launch_rank=%d omitted_or_delayed_rank=%d\n",o.rccl_scenario.c_str(),o.rccl_first,other);
        candidate.propose(input,1,1);
        if(missing)throw std::runtime_error("RCCL missing peer unexpectedly produced a successful proposal");
        exact_hybrid_layers(c,old,candidate);
        old.commit(1);candidate.commit(1);exact_hybrid_layers(c,old,candidate);
        const auto next=signed_input(H*N,7902,0.55f);
        old.propose(next,1,2);candidate.propose(next,1,2);exact_hybrid_layers(c,old,candidate);
        old.commit(1);candidate.commit(1);exact_hybrid_layers(c,old,candidate);
    }
    std::printf("RCCL_LAYER_FAULT_PASS scenario=delayed-rank first=%d delay_ms=%d exact_continuation=1 timing_admitted=0\n",o.rccl_first,o.rccl_delay_ms);
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

struct BenchStudy {
    tp::TpGdnLayer& baseline;tp::TpGdnLayer& candidate;
    tp::TpGdnExecution baseline_mode,candidate_mode;
    const char* baseline_label;const char* candidate_label;
    bool exact_transport=false;
};
void sustained_benchmark(Checks& c,const Options& o,const BenchStudy& study,const tp::TpGdnRankWeights& oracle,
                         uint64_t& epoch,const std::vector<float>& state,const std::vector<float>& conv) {
    auto& reference=study.baseline;auto& parallel=study.candidate;const auto mode=study.candidate_mode;
    const bool association=mode==tp::TpGdnExecution::ColumnCaptured;
    const bool hybrid=mode==tp::TpGdnExecution::HybridCaptured||mode==tp::TpGdnExecution::HybridRcclCaptured;
    std::printf("BLOCK_STUDY baseline=%s candidate=%s\n",study.baseline_label,study.candidate_label);
    std::printf("BLOCK_SCOPE complete-call evolving-state blocks; shadow pass checks every step, measured blocks download no intermediate outputs/state and check final results only; not full-model throughput\n");
    std::printf("BLOCK_METHOD calls=%d crossovers=%d same preuploaded input sequence in both orders; equal untimed %d-call evolving warm block immediately before each measured arm; initial state reset outside clock; block wall includes all host looping/proposals/commits/sync; no snapshots within block\n",o.block_calls,o.block_trials,o.block_calls);
    for(int tokens:{1,2,4,5,8}){
        reference.set_execution(study.baseline_mode);parallel.set_execution(study.candidate_mode);execution_metadata(study.baseline_mode);execution_metadata(study.candidate_mode);
        const size_t stride=size_t(tokens)*H*N;std::vector<float> sequence_values;
        for(int j=0;j<o.block_calls;++j){const auto v=signed_input(stride,3301+tokens*71+j*37,0.65f);sequence_values.insert(sequence_values.end(),v.begin(),v.end());}
        Buffer<float> input0(sequence_values.size(),0),input1(sequence_values.size(),1);input0.set(sequence_values);input1.set(sequence_values);
        for(int d=0;d<2;++d){on(d);ck(cudaDeviceSynchronize(),"block input upload completion");}
        auto input=[&](int j){return std::array<const float*,2>{input0.p+size_t(j)*stride,input1.p+size_t(j)*stride};};
        reference.reset_state(state,conv);parallel.reset_state(state,conv);
        tp::TpGdnLayerSnapshot shadow_full,shadow_tp;
        for(int j=0;j<o.block_calls;++j){const auto e=++epoch;reference.propose_device(input(j),tokens,e);reference.commit(tokens);parallel.propose_device(input(j),tokens,e);parallel.commit(tokens);
            shadow_full=reference.snapshot();shadow_tp=parallel.snapshot();
            std::printf("BLOCK_SHADOW mode=%s T=%d step=%d\n",execution_name(mode),tokens,j);
            seams(c,shadow_full,shadow_tp,false,true,association,&oracle,hybrid);seams(c,shadow_tp,parallel.snapshot(1),true);
            if(study.exact_transport)exact_hybrid_layers(c,reference,parallel);
            if(c.failures)throw std::runtime_error("sustained shadow parity failed; timing prohibited");}
        auto run=[&](tp::TpGdnLayer& layer){
            layer.reset_state(state,conv);const auto start=std::chrono::steady_clock::now();
            for(int j=0;j<o.block_calls;++j){layer.propose_device(input(j),tokens,++epoch);layer.commit(tokens);}
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        };
        std::vector<double> full_times,tp_times,ratios;
        for(int trial=0;trial<o.block_trials;++trial){double fs=0,ps=0;
            for(int leg=0;leg<2;++leg){const bool tp_first=((trial+leg)&1)!=0;double a=0,b=0;
                auto arm=[&](tp::TpGdnLayer& layer){(void)run(layer);return run(layer);};
                if(tp_first){b=arm(parallel);a=arm(reference);}else{a=arm(reference);b=arm(parallel);}
                std::printf("BLOCK_SAMPLE baseline=%s candidate=%s mode=%s T=%d trial=%d crossover-leg=%d first=%s calls=%d baseline-wall-ms=%.6f candidate-wall-ms=%.6f baseline-ms-per-call=%.6f candidate-ms-per-call=%.6f\n",study.baseline_label,study.candidate_label,execution_name(mode),tokens,trial,leg,tp_first?study.candidate_label:study.baseline_label,o.block_calls,a,b,a/o.block_calls,b/o.block_calls);
                // Epoch counters differ across sequential complete blocks; inputs,
                // initial state, token count and number of transitions match.
                const auto aa=reference.snapshot(),bb=parallel.snapshot();seams(c,aa,bb,false,false,association,&oracle,hybrid);seams(c,bb,parallel.snapshot(1),true);
                seams(c,shadow_full,aa,false,false);seams(c,shadow_tp,bb,false,false);
                if(study.exact_transport)for(int rank=0;rank<2;++rank)exact_hybrid(c,reference.snapshot(rank),parallel.snapshot(rank),false);
                if(c.failures)throw std::runtime_error("measured block final parity failed; timing rejected");
                fs+=a;ps+=b;full_times.push_back(a/o.block_calls);tp_times.push_back(b/o.block_calls);
            }
            ratios.push_back(fs/ps);std::printf("BLOCK_CROSSOVER baseline=%s candidate=%s mode=%s T=%d trial=%d baseline-mean-ms-per-call=%.6f candidate-mean-ms-per-call=%.6f crossed-ratio=%.6f\n",study.baseline_label,study.candidate_label,execution_name(mode),tokens,trial,fs/(2*o.block_calls),ps/(2*o.block_calls),fs/ps);
        }
        std::printf("BLOCK_SUMMARY baseline=%s candidate=%s mode=%s T=%d blocks-per-arm=%zu calls-per-block=%d baseline-ms-per-call-median=%.6f candidate-ms-per-call-median=%.6f crossover-ratio-median=%.6f ratio-of-total-times=%.6f measured-intermediate-parity=not-downloaded shadow-every-step=passed final-block-parity=passed\n",study.baseline_label,study.candidate_label,execution_name(mode),tokens,full_times.size(),o.block_calls,quantile(full_times,.5),quantile(tp_times,.5),quantile(ratios,.5),std::accumulate(full_times.begin(),full_times.end(),0.0)/std::accumulate(tp_times.begin(),tp_times.end(),0.0));
    }
}
void benchmark(Checks& c,const Options& o,const BenchStudy& study,const tp::TpGdnRankWeights& oracle,
               uint64_t& epoch,const std::vector<float>& state,const std::vector<float>& conv) {
    auto& reference=study.baseline;auto& parallel=study.candidate;const auto mode=study.candidate_mode;
    const bool association=mode==tp::TpGdnExecution::ColumnCaptured;
    const bool hybrid=mode==tp::TpGdnExecution::HybridCaptured||mode==tp::TpGdnExecution::HybridRcclCaptured;
    std::printf("BENCH_STUDY baseline=%s candidate=%s\n",study.baseline_label,study.candidate_label);
    std::printf("\nBENCH_SCOPE real-weight-one-GDN-layer; explicit-arm-labels-above; matched-input-and-state; not-EP-or-full-model-generation\n");
    std::printf("BENCH_METHOD wall=propose_device+commit(T), includes D2D input/host launch/P2P/compute/commit/sync; excludes reset/H2D/weight-load/graph-preparation/snapshots; fixed initialized state, varying matched signed input; keep=T is a workload choice, not measured speculative acceptance\n");
    std::printf("BENCH_CONTRACT both arms use speculative proposal plus explicit commit replay, including T=1; this is not optimized autoregressive self-commit or the production Verifier/EP baseline; fixed native-GR/native-BF16-shared/exact-MMVQ settings and explicit expert mode=%d\n",o.mode);
    std::printf("BENCH_LIMIT execution modes run in separate blocks; only each named study is directly paired, so cross-study absolute timing differences are diagnostic and may include thermal/clock drift\n");
    std::printf("BENCH_CONFIG warmup_crossovers=%d measured_crossovers=%d pairs-per-crossover=2 order=ABBA/BAAB-alternating same-input-both-orders; one untimed warm call per arm before each measured pair; snapshots after each pair outside timer, checks after crossover; summaries conditional on all gates passing\n",o.warmup,o.trials);
    // Prepare each graph once, then reuse across later T changes. Preparation is
    // never part of a timing sample and has no claimed amortization horizon.
    for(int tokens:{1,2,4,5,8}){
        reference.set_execution(study.baseline_mode);parallel.set_execution(study.candidate_mode);execution_metadata(study.baseline_mode);execution_metadata(study.candidate_mode);
        Buffer<float>input0(size_t(tokens)*H*N,0),input1(size_t(tokens)*H*N,1);
        std::vector<double>rt,pt,ratio,rp,pp,rc,pc;
        for(int trial=-o.warmup;trial<o.trials;++trial){
            const auto input=signed_input(size_t(tokens)*H*N,1201+tokens*97+(trial+o.warmup)*31,0.65f);
            input0.set(input);input1.set(input);
            // Pageable/default-stream uploads do not establish a dependency on
            // either executor's nonblocking compute stream. Finish both outside
            // all clocks, rather than rely on an incidental reset-state copy.
            for(int d=0;d<2;++d){on(d);ck(cudaDeviceSynchronize(),"benchmark input upload completion");}
            const std::array<const float*,2> inputs{input0.p,input1.p};
            const bool start_tp=((trial+o.warmup)&1)!=0;
            auto one=[&](tp::TpGdnLayer& layer,uint64_t e){layer.reset_state(state,conv);return timed_layer(layer,inputs,tokens,e);};
            double full_sum=0,tp_sum=0;
            std::array<tp::TpGdnLayerSnapshot,2> full_snap,split_snap,replica_snap,baseline_replica_snap;
            for(int order=0;order<2;++order){
                if(trial>=0){
                    // Refresh both cards after out-of-band diagnostics. This is
                    // a reset-state layer experiment, not uninterrupted decode.
                    const auto warm_epoch=++epoch;const bool warm_tp=start_tp^(order!=0);
                    if(warm_tp){(void)one(parallel,warm_epoch);(void)one(reference,warm_epoch);}
                    else{(void)one(reference,warm_epoch);(void)one(parallel,warm_epoch);}
                }
                const uint64_t e=++epoch;const bool tp_first=start_tp^(order!=0);LayerTime a,b;
                if(tp_first){b=one(parallel,e);a=one(reference,e);}else{a=one(reference,e);b=one(parallel,e);}
                std::printf("BENCH_%s baseline=%s candidate=%s mode=%s T=%d trial=%d crossover-leg=%d first=%s baseline-propose-ms=%.6f baseline-commit-ms=%.6f baseline-total-ms=%.6f candidate-propose-ms=%.6f candidate-commit-ms=%.6f candidate-total-ms=%.6f\n",trial<0?"WARMUP":"SAMPLE",study.baseline_label,study.candidate_label,execution_name(mode),tokens,trial,order,tp_first?study.candidate_label:study.baseline_label,a.propose_ms,a.commit_ms,a.total_ms,b.propose_ms,b.commit_ms,b.total_ms);
                full_snap[order]=reference.snapshot();split_snap[order]=parallel.snapshot();replica_snap[order]=parallel.snapshot(1);
                if(study.exact_transport)baseline_replica_snap[order]=reference.snapshot(1);
                full_sum+=a.total_ms;tp_sum+=b.total_ms;
                if(trial>=0){rt.push_back(a.total_ms);pt.push_back(b.total_ms);rp.push_back(a.propose_ms);pp.push_back(b.propose_ms);rc.push_back(a.commit_ms);pc.push_back(b.commit_ms);}
            }
            // Every measured proposal/commit pair is retained and checked,
            // including the first order leg. Snapshot gaps are outside clocks;
            // explicit untimed warming precedes the next measured pair.
            for(int order=0;order<2;++order){seams(c,full_snap[order],split_snap[order],false,true,association,&oracle,hybrid);seams(c,split_snap[order],replica_snap[order],true);
                if(study.exact_transport){exact_hybrid(c,full_snap[order],split_snap[order]);exact_hybrid(c,baseline_replica_snap[order],replica_snap[order]);}}
            if(c.failures)throw std::runtime_error("benchmark parity failed; timing is not an accepted result");
            if(trial>=0)std::printf("BENCH_CROSSOVER baseline=%s candidate=%s mode=%s T=%d trial=%d baseline-mean-ms=%.6f candidate-mean-ms=%.6f crossed-ratio=%.6f\n",study.baseline_label,study.candidate_label,execution_name(mode),tokens,trial,full_sum/2,tp_sum/2,full_sum/tp_sum);
            if(trial>=0)ratio.push_back(full_sum/tp_sum);
        }
        std::printf("BENCH_SUMMARY baseline=%s candidate=%s mode=%s T=%d pairs=%zu baseline-ms-median=%.6f baseline-ms-p10=%.6f baseline-ms-p90=%.6f candidate-ms-median=%.6f candidate-ms-p10=%.6f candidate-ms-p90=%.6f crossover-ratio-median=%.6f ratio-of-total-times=%.6f baseline-propose-median=%.6f candidate-propose-median=%.6f baseline-commit-median=%.6f candidate-commit-median=%.6f\n",study.baseline_label,study.candidate_label,execution_name(mode),tokens,rt.size(),quantile(rt,.5),quantile(rt,.1),quantile(rt,.9),quantile(pt,.5),quantile(pt,.1),quantile(pt,.9),quantile(ratio,.5),std::accumulate(rt.begin(),rt.end(),0.0)/std::accumulate(pt.begin(),pt.end(),0.0),quantile(rp,.5),quantile(pp,.5),quantile(rc,.5),quantile(pc,.5));
        std::fflush(stdout);
    }
}
void calibration(Checks& c,const Options& o,const strata::core::ModelGeometry& g,
                 tp::TpGdnLayer& full0,tp::TpGdnLayer& row,const tp::TpGdnRankWeights& oracle,
                 uint64_t& epoch,const std::vector<float>& state,const std::vector<float>& conv) {
    tp::TpGdnWeights weights1;std::string error;
    if(!weights1.load(o.shards,o.pack,g,o.layer,-1,1,error))throw std::runtime_error("calibration GPU1 full load: "+error);
    tp::TpGdnLayer full1(g,weights1.weights(),nullptr,8,o.mode);
    for(int t:{1,4,8})full1.prepare_captured(t);
    full0.set_execution(tp::TpGdnExecution::Captured);full1.set_execution(tp::TpGdnExecution::Captured);row.set_execution(tp::TpGdnExecution::Captured);
    std::printf("CAL_SCOPE version=1 partition=output-rows T=1,4,8 phase-input=frozen-valid-own-trajectory half-input=identical-isolated-concurrent full-input=same-initial-residual-state full-phase-input=own-valid-trajectory phase-wall=launch-event-submit-stream-sync device-span=rank-local-own-graph-or-explicit-peer-join phase-additive=0 full-minus-phase=forbidden snapshot-restore=all-mutable-H2D-outside-timer cache=restore-conditioned calls=single-not-sustained idle-rank-sync=included weights=real hc-router=coarse-combined clocks-power=read-only-runner\n");
    for(int tokens:{1,4,8}){
        const auto input=signed_input(size_t(tokens)*H*N,2401+tokens*37,0.65f);
        Buffer<float> input0(input.size(),0),input1(input.size(),1);input0.set(input);input1.set(input);
        std::array<tp::TpGdnLayer*,3> layers{&full0,&full1,&row};
        // Both full-device baselines are tested against row TP, with the same
        // input, checkpoint and accept-all commit. Reset/download outside clocks.
        for(int trial=-o.warmup;trial<o.trials;++trial)for(int device=0;device<2;++device){
            const int d=(trial&1)?1-device:device;
            std::array<tp::TpGdnLayerSnapshot,4> shots;
            for(int order=0;order<4;++order){
                const bool candidate=(order==1||order==2)^bool(trial&1);
                auto& layer=candidate?row:*layers[d];layer.reset_state(state,conv);
                const std::array<const float*,2> pointers=candidate?std::array<const float*,2>{input0.p,input1.p}:
                    std::array<const float*,2>{d?input1.p:input0.p,nullptr};
                const auto timing=timed_layer(layer,pointers,tokens,++epoch);shots[order]=layer.snapshot();
                if(candidate)seams(c,shots[order],layer.snapshot(1),true);
                if(trial>=0)std::printf("CAL_WALL T=%d trial=%d baseline-device=%d order=%d arm=%s propose-ms=%.6f commit-ms=%.6f total-ms=%.6f scope=propose-device-commit-accept-all\n",tokens,trial,d,order,candidate?"row-tp":"full",timing.propose_ms,timing.commit_ms,timing.total_ms);
            }
            const int a=(trial&1)?1:0,b=(trial&1)?0:1;
            seams(c,shots[a],shots[b],false,false,false,&oracle);seams(c,shots[3-a],shots[3-b],false,false,false,&oracle);
            c.floats("calibration committed state",shots[a].state,shots[b].state,gdn_gate);
            c.floats("calibration committed conv",shots[a].conv,shots[b].conv,gdn_gate);
            c.floats("calibration second state",shots[3-a].state,shots[3-b].state,gdn_gate);
            c.floats("calibration second conv",shots[3-a].conv,shots[3-b].conv,gdn_gate);
            if(c.failures)throw std::runtime_error("calibration whole-layer parity failed; samples invalid");
        }
        // Device ordering alternates by T. Full0/full1 and both row halves have
        // identical initial residual/checkpoint; within-phase repeat inputs are
        // byte-identical. No claim that full and row intermediate roundoff agrees.
        for(int index:tokens==4?std::array<int,3>{1,0,2}:std::array<int,3>{0,1,2}){
            auto& layer=*layers[index];layer.reset_state(state,conv);
            const auto samples=layer.calibrate_frozen(input,tokens,o.warmup,o.trials);
            for(const auto& x:samples)std::printf("CAL_PHASE T=%d owner=%s device0=%d device1=%d phase=%s arm=%s trial=%d order=%d calls=%d restore-H2D-bytes=%llu peer-bytes-per-rank=%llu wall-ms=%.6f rank0-event-ms=%.6f rank1-event-ms=%.6f event-scope=%s replay-output=%s\n",tokens,index==2?"row-tp":"full",layer.device(0),layer.ranks()==2?layer.device(1):-1,x.phase.c_str(),x.arm.c_str(),x.trial,x.order,x.calls,(unsigned long long)x.restored_bytes,(unsigned long long)x.peer_bytes_per_rank,x.wall_ms,x.device_ms[0],x.device_ms[1],x.event_scope.c_str(),x.arm.find("empty")!=std::string::npos?"not-applicable":"exact");
            std::fflush(stdout);
        }
    }
    std::printf("CAL_GATE PASS whole-layer-parity=checked frozen-compute-output=exact column=excluded scope=one-layer-no-full-model-speedup-claim\n");
}

uint64_t fixture_hash(const std::vector<float>& values) {
    // Stable byte fingerprint of exactly the supplied float buffer. This is an
    // identity aid, not a cryptographic hash or evidence of model authenticity.
    uint64_t h=14695981039346656037ull;
    for(const auto* p=reinterpret_cast<const uint8_t*>(values.data());
        p!=reinterpret_cast<const uint8_t*>(values.data()+values.size());++p){h^=*p;h*=1099511628211ull;}
    return h;
}
void probe_weight_metadata(const char* owner,const tp::TpGdnRankWeights& w) {
    const auto& gu=w.gu_layout;const auto& down=w.down_layout;
    std::printf("PROBE_NATIVE owner=%s rank=%d device=%d gu-type=%s gu-type-id=%d gu-input=%lld gu-hidden=%lld gu-row-bytes=%zu down-type=%s down-type-id=%d down-input=%lld down-output=%lld down-row-bytes=%zu\n",owner,w.rank,w.device,strata::ggml_type_name(gu.gu_type),gu.gu_type,(long long)gu.n_embd,(long long)gu.n_ff,gu.gu_row,strata::ggml_type_name(down.d_type),down.d_type,(long long)down.n_ff,(long long)down.n_embd,down.d_row);
    for(const auto& item:std::array<std::pair<const char*,const tp::TpNativeMatrix*>,6>{{
        {"qkv",&w.qkv},{"z",&w.z},{"attention-output",&w.out},{"shared-gate",&w.shared_gate},{"shared-up",&w.shared_up},{"shared-down",&w.shared_down}}}){
        const auto& m=*item.second;
        std::printf("PROBE_MATRIX owner=%s matrix=%s type=%s type-id=%d input=%d output=%d\n",owner,item.first,strata::ggml_type_name(m.type),m.type,m.input,m.output);
    }
}
void model_probe(const Options& o) {
    strata::core::ModelGeometry g;std::string error;
    tp::TpGdnWeights weights0,weights1,hybrid0,hybrid1;
    if(!weights0.load(o.shards,o.pack,g,o.layer,-1,0,error))throw std::runtime_error("probe full GPU0 load: "+error);
    if(!weights1.load(o.shards,o.pack,g,o.layer,-1,1,error))throw std::runtime_error("probe full GPU1 load: "+error);
    if(!hybrid0.load(o.shards,o.pack,g,o.layer,0,0,error,tp::TpGdnPartition::HybridRowsColumns))throw std::runtime_error("probe hybrid rank0 load: "+error);
    if(!hybrid1.load(o.shards,o.pack,g,o.layer,1,1,error,tp::TpGdnPartition::HybridRowsColumns))throw std::runtime_error("probe hybrid rank1 load: "+error);
    std::printf("PROBE_SCOPE version=1 layer=%d mode=%d T=1,8 attention=output-rows FFN=local-hidden-input-columns down-K-full=640 down-K-half=320 HC=replicated-full-BF16 warmup=%d trials=%d phase-additive=0 timing=diagnostic-only production-defaults=unchanged\n",o.layer,o.mode,o.warmup,o.trials);
    std::printf("PROBE_WEIGHTS full0-bytes=%llu full1-bytes=%llu hybrid0-bytes=%llu hybrid1-bytes=%llu sources=supplied-real-layer only-selected-layer-no-model-generation\n",(unsigned long long)weights0.weight_bytes(),(unsigned long long)weights1.weight_bytes(),(unsigned long long)hybrid0.weight_bytes(),(unsigned long long)hybrid1.weight_bytes());
    std::printf("PROBE_CONTRACT frozen-seams=canonical-full0 full-vs-half=coordinated isolated-vs-concurrent=identical-bytes restore=outside-timer restore-order=inactive-first-active-last cache=restore-conditioned calls=single-not-sustained GPU-event-clocks=rank-local event-spans=adjacent-graph-including-host-enqueue-gap no-full-minus-phase-arithmetic no-speedup-from-summed-phases concurrent=scheduled actual-overlap=not-verified down-production-comparison=shape-and-kernel-family-change production-EP-inference=out-of-scope\n");
    probe_weight_metadata("full0",weights0.weights());probe_weight_metadata("full1",weights1.weights());
    probe_weight_metadata("half0",hybrid0.weights());probe_weight_metadata("half1",hybrid1.weights());
    for(const char* flag:{"STRATA_HC_PERSIST","STRATA_HC_SPLIT","STRATA_GR_FAST","STRATA_GR_DOWN_MAX4","STRATA_NO_MULTI_GR",
                         "STRATA_OLD_IQ_MMVQ","STRATA_NO_SUB16_GU","STRATA_GROUPED_V1","STRATA_IQ_STAGE_GRID","STRATA_IQ_STAGE_GRID_MMVQ",
                         "STRATA_EXPERT_V2","STRATA_EXPERT_V2K","STRATA_TSUM","STRATA_HIP_SWIGLU_FUSED","STRATA_EXP_MODE",
                         "STRATA_MMVQ_WAVE","STRATA_NO_MMVQ_ROWS2","STRATA_MMVQ_IL_ROWS","STRATA_Q8_PACKED","STRATA_Q6_PACKED"}){
        const char* value=std::getenv(flag);std::printf("PROBE_ENV %s=%s\n",flag,value?value:"unset");
    }
    for(int d=0;d<2;++d){on(d);std::printf("PROBE_DISPATCH device=%d hc-variant=%d hc-projections=BF16 hc-norm=F32 hc-down-rows=320 hc-streams=4 native-gr=1 native-shared-bf16=1 mmvq-multi-exact=1 expert-mode=%d\n",d,k::fused_gr_variant(),o.mode);}
    Checks c;coherence(c,g,weights0.weights(),hybrid0.weights());coherence(c,g,weights0.weights(),hybrid1.weights());
    if(c.failures)throw std::runtime_error("probe actual weight coherence gate failed; no timing admitted");
    tp::TpGdnLayer full0(g,weights0.weights(),nullptr,8,o.mode),full1(g,weights1.weights(),nullptr,8,o.mode),
        hybrid(g,hybrid0.weights(),&hybrid1.weights(),8,o.mode);
    const std::array<tp::TpGdnLayer*,3> layers{&full0,&full1,&hybrid};
    for(auto* layer:layers)for(int t:{1,8})layer->prepare_captured(t);
    full0.set_execution(tp::TpGdnExecution::Captured);full1.set_execution(tp::TpGdnExecution::Captured);
    hybrid.set_execution(tp::TpGdnExecution::HybridCaptured);
    const auto state=signed_input(STATE,17,0.015f),conv=signed_input(CONV,31,0.12f);
    uint64_t epoch=0;int cases=0;
    std::printf("PROBE_CORRECTNESS scope=targeted-not-all44-prefixes cases=T1-keep0,T1-keep1,T8-keep0,T8-keep4,T8-keep8 continuation-T=1 gates=unchanged-original-reference-and-independent-legacy full-GPU1=checked\n");
    for(const auto& spec:std::array<std::array<int,2>,5>{{{{1,0}},{{1,1}},{{8,0}},{{8,4}},{{8,8}}}}){
        const int tokens=spec[0],keep=spec[1];
        const auto input=signed_input(size_t(tokens)*H*N,2401+tokens*37,0.65f);
        std::printf("PROBE_CASE T=%d keep=%d epoch=%llu residual-fnv1a64=%016llx state-fnv1a64=%016llx conv-fnv1a64=%016llx\n",tokens,keep,(unsigned long long)(epoch+1),(unsigned long long)fixture_hash(input),(unsigned long long)fixture_hash(state),(unsigned long long)fixture_hash(conv));
        ++epoch;for(auto* layer:layers){layer->reset_state(state,conv);layer->propose(input,tokens,epoch);}
        auto a=full0.snapshot(),b=hybrid.snapshot();
        seams(c,a,full1.snapshot());seams(c,a,b,false,true,false,&weights0.weights(),true);seams(c,b,hybrid.snapshot(1),true);
        c.exact("probe proposal leaves state",state,b.state);c.exact("probe proposal leaves conv",conv,b.conv);
        route_margins(c,weights0.weights(),a);
        const auto legacy=legacy_gdn(weights0.weights(),a.attention_input,state,conv,tokens,keep);
        c.floats("probe serial legacy GDN",legacy.output,b.gdn_output,gdn_gate);
        c.floats("probe serial legacy mixer",legacy.mixer,b.mixer_output,gdn_gate);
        legacy_hc(c,weights0.weights(),input,b);
        for(auto* layer:layers)layer->commit(keep);
        committed(c,full0.snapshot(),hybrid.snapshot(),legacy);committed(c,full0.snapshot(),full1.snapshot(),legacy);
        // One-token continuation keeps the probe limited to T1/T8 and catches
        // stale banks, reject-all rollback, and partial-prefix tail leakage.
        const auto next=signed_input(size_t(H)*N,5001+tokens*13+keep*23,0.55f);
        ++epoch;for(auto* layer:layers)layer->propose(next,1,epoch);
        a=full0.snapshot();b=hybrid.snapshot();
        seams(c,a,full1.snapshot());seams(c,a,b,false,true,false,&weights0.weights(),true);seams(c,b,hybrid.snapshot(1),true);
        const auto continuation=legacy_gdn(weights0.weights(),a.attention_input,legacy.state,legacy.conv,1,1);
        c.floats("probe continuation GDN",continuation.output,b.gdn_output,gdn_gate);
        c.floats("probe continuation mixer",continuation.mixer,b.mixer_output,gdn_gate);
        for(auto* layer:layers)layer->commit(1);
        committed(c,full0.snapshot(),hybrid.snapshot(),continuation);committed(c,full0.snapshot(),full1.snapshot(),continuation);
        ++cases;std::fflush(stdout);
        if(c.failures)throw std::runtime_error("probe targeted parity failed; no timing admitted");
    }
    std::printf("PROBE_CORRECTNESS_GATE PASS cases=%d continuation-cases=%d failures=%d coverage=bounded-not-all-prefixes\n",cases,cases,c.failures);
    for(int tokens:{1,8}){
        const auto input=signed_input(size_t(tokens)*H*N,2401+tokens*37,0.65f);
        for(auto* layer:layers)layer->reset_state(state,conv);
        std::printf("PROBE_FIXTURE T=%d residual-fnv1a64=%016llx state-fnv1a64=%016llx conv-fnv1a64=%016llx input-kind=synthetic-signed-activations-real-weights no-prompt-or-logits\n",tokens,(unsigned long long)fixture_hash(input),(unsigned long long)fixture_hash(state),(unsigned long long)fixture_hash(conv));
        const auto samples=tp::TpGdnLayer::calibrate_fine_frozen(full0,full1,hybrid,input,tokens,o.warmup,o.trials);
        for(const auto& x:samples){
            std::string entries;for(int n:x.group_entries){if(!entries.empty())entries+=",";entries+=std::to_string(n);}
            if(entries.empty())entries="none";
            std::printf("PROBE_PHASE T=%d owner=%s device0=%d device1=%d phase=%s arm=%s trial=%d order=%d calls=%d input-fnv1a64=%016llx groups=%d group-entries=%s restore-H2D-bytes=%llu peer-bytes-per-rank=%llu wall-ms=%.6f rank0-event-ms=%.6f rank1-event-ms=%.6f event-scope=%s\n",tokens,x.owner.c_str(),x.devices[0],x.devices[1],x.phase.c_str(),x.arm.c_str(),x.trial,x.order,x.calls,(unsigned long long)x.input_hash,x.groups,entries.c_str(),(unsigned long long)x.restored_bytes,(unsigned long long)x.peer_bytes_per_rank,x.wall_ms,x.device_ms[0],x.device_ms[1],x.event_scope.c_str());
        }
        std::fflush(stdout);
    }
    std::printf("PROBE_GATE PASS targeted-whole-layer-parity=checked frozen-replay-gates=checked full-model-correctness=unmeasured full-model-throughput=unmeasured\n");
}
} // namespace
int main(int argc,char** argv) {
    try {
        const auto o=parse(argc,argv);if(o.model_probe||o.execution=="hybrid-rccl")std::setvbuf(stdout,nullptr,_IOLBF,0);hc_source_metadata(o);int devices=0;ck(cudaGetDeviceCount(&devices),"device count");
        if(devices<2)throw std::runtime_error("two GPUs required; this is not a CPU or single-GPU pass");
        for(int d=0;d<2;++d){cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,d),"device properties");std::printf("device=%d name=%s\n",d,prop.name);int access=0;ck(cudaDeviceCanAccessPeer(&access,d,1-d),"P2P check");if(!access)throw std::runtime_error("bidirectional P2P required");}
        for(const char* flag:{"STRATA_GR_V3","STRATA_GR_SPLIT"}){
            const char* value=std::getenv(flag);
            if(value&&std::atoi(value)!=0)throw std::runtime_error(std::string("parity gate requires ")+flag+"=0 before process startup");
            std::printf("HC_DISPATCH %s=%s effective=0\n",flag,value?value:"unset");
        }
        k::gr_set_native_mmvf(true);k::shared_expert_set_native_bf16(true);k::native_mmvq_set_multi_exact(true);k::native_expert_set_mode(o.mode,0);
        if(o.model_probe){model_probe(o);return 0;}
        if(o.execution=="hybrid-rccl"&&o.rccl_scenario!="normal"){rccl_fault_gate(o);return 0;}
        const auto executions=execution_modes(o);
        const bool use_flat=std::find(executions.begin(),executions.end(),tp::TpGdnExecution::FlatCaptured)!=executions.end()||
                            std::find(executions.begin(),executions.end(),tp::TpGdnExecution::FlatHcCaptured)!=executions.end();
        Checks preflight;if(use_flat)flat_protocol_preflight(preflight);
        const bool use_column=std::find(executions.begin(),executions.end(),tp::TpGdnExecution::ColumnCaptured)!=executions.end();
        const bool use_hybrid=std::find(executions.begin(),executions.end(),tp::TpGdnExecution::HybridCaptured)!=executions.end();
        const bool use_rccl=o.execution=="hybrid-rccl";
        const bool profiles=(o.profile_flat||o.benchmark)&&!use_rccl;
        strata::core::ModelGeometry g;tp::TpGdnWeights full,rank0,rank1,column0,column1,hybrid0,hybrid1;std::string err;
        if(!full.load(o.shards,o.pack,g,o.layer,-1,0,err))throw std::runtime_error("full reference load: "+err);
        if(!rank0.load(o.shards,o.pack,g,o.layer,0,0,err))throw std::runtime_error("rank0 load: "+err);
        if(!rank1.load(o.shards,o.pack,g,o.layer,1,1,err))throw std::runtime_error("rank1 load: "+err);
        if(use_column){
            if(!column0.load(o.shards,o.pack,g,o.layer,0,0,err,tp::TpGdnPartition::InputColumns))throw std::runtime_error("column rank0 load: "+err);
            if(!column1.load(o.shards,o.pack,g,o.layer,1,1,err,tp::TpGdnPartition::InputColumns))throw std::runtime_error("column rank1 load: "+err);
            std::printf("column-weight-bytes rank0=%llu rank1=%llu\n",(unsigned long long)column0.weight_bytes(),(unsigned long long)column1.weight_bytes());
        }
        if(use_hybrid){
            if(!hybrid0.load(o.shards,o.pack,g,o.layer,0,0,err,tp::TpGdnPartition::HybridRowsColumns))throw std::runtime_error("hybrid rank0 load: "+err);
            if(!hybrid1.load(o.shards,o.pack,g,o.layer,1,1,err,tp::TpGdnPartition::HybridRowsColumns))throw std::runtime_error("hybrid rank1 load: "+err);
            std::printf("hybrid-weight-bytes rank0=%llu rank1=%llu\n",(unsigned long long)hybrid0.weight_bytes(),(unsigned long long)hybrid1.weight_bytes());
        }
        std::printf("real-weight GDN layer=%d mode=%d weight-bytes full=%llu rank0=%llu rank1=%llu\n",o.layer,o.mode,(unsigned long long)full.weight_bytes(),(unsigned long long)rank0.weight_bytes(),(unsigned long long)rank1.weight_bytes());
        std::printf("scope=one-layer; full-reference shares orchestrator; independent legacy GDN/HC/shared and CPU-combine checks; no model-quality or performance claim\n");
        Checks c;coherence(c,g,full.weights(),rank0.weights());coherence(c,g,full.weights(),rank1.weights());
        if(use_column){coherence(c,g,full.weights(),column0.weights());coherence(c,g,full.weights(),column1.weights());}
        if(use_hybrid){coherence(c,g,full.weights(),hybrid0.weights());coherence(c,g,full.weights(),hybrid1.weights());}
        if(c.failures)return 1;
        if(use_hybrid)std::printf("HYBRID_REFERENCE_CONTRACT original-full bounds unchanged; original FFN input/router logits/router weights/ordered IDs/hidden Q8 exact; candidate-same-input full-FFN oracle retained; local fused-vs-unfused FFN gate; GDN output row ownership and hidden column ownership checked independently; correctness failures prohibit timing\n");
        if(use_column)std::printf("COLUMN_REFERENCE_CONTRACT original-full seam/residual/state bounds unchanged; original router IDs exact; router logits/weights use predeclared hc_gate(abs2e-6,scaled2e-5,rms5e-6) only when FFN inputs differ; candidate-same-input original full GU/Q8 exact and complete FFN numerical gate; original-input Q8 differences diagnostic, not exact PASS; diagnostic recomputed expert parts do not establish fused output correctness; engineering parity only\n");
        const auto state=signed_input(STATE,17,0.015f),conv=signed_input(CONV,31,0.12f);
        if(use_flat)flat_failure_gate(c,o,g,rank0.weights(),rank1.weights(),state,conv);
        if(use_rccl)rccl_lifecycle_gate(c,o,g,hybrid0.weights(),hybrid1.weights(),state,conv);
        tp::TpGdnLayer reference(g,full.weights(),nullptr,8,o.mode),row_parallel(g,rank0.weights(),&rank1.weights(),8,o.mode);
        std::unique_ptr<tp::TpGdnLayer> column_parallel,hybrid_parallel,rccl_parallel;
        if(use_column)column_parallel=std::make_unique<tp::TpGdnLayer>(g,column0.weights(),&column1.weights(),8,o.mode);
        if(use_hybrid)hybrid_parallel=std::make_unique<tp::TpGdnLayer>(g,hybrid0.weights(),&hybrid1.weights(),8,o.mode);
        if(use_rccl)rccl_parallel=std::make_unique<tp::TpGdnLayer>(g,hybrid0.weights(),&hybrid1.weights(),8,o.mode);
        auto candidate_for=[&](tp::TpGdnExecution mode)->tp::TpGdnLayer&{
            if(mode==tp::TpGdnExecution::ColumnCaptured)return *column_parallel;
            if(mode==tp::TpGdnExecution::HybridCaptured)return *hybrid_parallel;
            if(mode==tp::TpGdnExecution::HybridRcclCaptured)return *rccl_parallel;
            return row_parallel;
        };
        for(auto execution:executions){
            auto& candidate=candidate_for(execution);
            if(execution==tp::TpGdnExecution::Captured||execution==tp::TpGdnExecution::ColumnCaptured||execution==tp::TpGdnExecution::HybridCaptured){
                for(int t=1;t<=8;++t){candidate.prepare_captured(t);if(profiles)candidate.prepare_captured(t,true);}
                if(o.benchmark||o.calibrate)for(int t:{1,2,4,5,8})reference.prepare_captured(t);
            }
            if(execution==tp::TpGdnExecution::HybridRcclCaptured)for(int t=1;t<=8;++t)candidate.prepare_rccl(t,o.rccl);
            if(execution==tp::TpGdnExecution::FlatCaptured||execution==tp::TpGdnExecution::FlatHcCaptured){
                const bool shard_hc=execution==tp::TpGdnExecution::FlatHcCaptured;
                for(int t=1;t<=8;++t){candidate.prepare_flat(t,shard_hc,false);if(profiles)candidate.prepare_flat(t,shard_hc,true);}
                if(o.benchmark)for(int t:{1,2,4,5,8})reference.prepare_flat(t,shard_hc,false);
            }
        }
        uint64_t epoch=0;int cases=0,rccl_cases=0;
        std::vector<int32_t> prior_routes;bool routes_changed=false,repeated_ids=false;
        for(auto execution:executions){
        const bool rccl=execution==tp::TpGdnExecution::HybridRcclCaptured;
        const bool association=execution==tp::TpGdnExecution::ColumnCaptured;
        const bool hybrid=execution==tp::TpGdnExecution::HybridCaptured||execution==tp::TpGdnExecution::HybridRcclCaptured;
        auto& parallel=candidate_for(execution);
        const bool profiled=profiles&&(execution==tp::TpGdnExecution::Captured||association||hybrid||execution==tp::TpGdnExecution::FlatCaptured||execution==tp::TpGdnExecution::FlatHcCaptured);
        parallel.set_execution(execution,false);execution_metadata(execution);
        std::printf("\nCORRECTNESS_EXECUTION %s (reference=runtime)\n",execution_name(execution));
        for(int tokens=1;tokens<=8;++tokens)for(int keep=0;keep<=tokens;++keep){
            std::printf("\nCASE T=%d keep=%d epoch=%llu\n",tokens,keep,(unsigned long long)(epoch+1));std::fflush(stdout);
            const auto input=signed_input(size_t(tokens)*H*N,43+tokens*19+keep*7,0.65f);
            reference.reset_state(state,conv);parallel.reset_state(state,conv);
            if(rccl){hybrid_parallel->reset_state(state,conv);parallel.set_rccl_launch_for_test((tokens+keep)&1);}
            reference.propose(input,tokens,++epoch);parallel.propose(input,tokens,epoch);
            if(rccl){hybrid_parallel->propose(input,tokens,epoch);exact_hybrid_layers(c,*hybrid_parallel,parallel);}
            auto a=reference.snapshot(),b=parallel.snapshot();seams(c,a,b,false,true,association,&full.weights(),hybrid);seams(c,b,parallel.snapshot(1),true);
            c.exact("proposal leaves state",state,b.state);c.exact("proposal leaves conv",conv,b.conv);
            if(!prior_routes.empty()&&std::vector<int32_t>(b.ids.begin(),b.ids.begin()+K)!=prior_routes)routes_changed=true;
            prior_routes.assign(b.ids.begin(),b.ids.begin()+K);
            {std::array<bool,512> seen{};for(int id:b.ids){if(id<0||id>=512)throw std::runtime_error("invalid route ID");repeated_ids|=seen[id];seen[id]=true;}}
            std::printf("ROUTE_MARGIN_REFERENCE original-full\n");route_margins(c,full.weights(),a);
            if(association){std::printf("ROUTE_MARGIN_REFERENCE column-candidate\n");route_margins(c,full.weights(),b);}
            const auto legacy=legacy_gdn(full.weights(),a.attention_input,state,conv,tokens,keep);
            c.floats("serial legacy GDN output",legacy.output,b.gdn_output,gdn_gate);
            c.floats("serial legacy mixer output",legacy.mixer,b.mixer_output,gdn_gate);
            legacy_hc(c,full.weights(),input,b);if(!association&&!hybrid)same_input_ffn_oracle(c,full.weights(),b);
            reference.commit(keep);parallel.commit(keep);a=reference.snapshot();b=parallel.snapshot();committed(c,a,b,legacy);
            if(rccl){hybrid_parallel->commit(keep);exact_hybrid_layers(c,*hybrid_parallel,parallel);}
            // A new input after EVERY keep catches accidental full-window commit,
            // rejected-tail leakage, stale exchange slots and zero-keep rollback.
            const auto next=signed_input(size_t(2)*H*N,101+tokens*13+keep*23,0.55f);
            reference.propose(next,2,++epoch);parallel.propose(next,2,epoch);
            if(rccl){hybrid_parallel->propose(next,2,epoch);exact_hybrid_layers(c,*hybrid_parallel,parallel);}
            const auto na=reference.snapshot(),nb=parallel.snapshot();seams(c,na,nb,false,true,association,&full.weights(),hybrid);seams(c,nb,parallel.snapshot(1),true);
            const auto continuation=legacy_gdn(full.weights(),na.attention_input,legacy.state,legacy.conv,2,2);
            c.floats("continuation legacy GDN",continuation.output,nb.gdn_output,gdn_gate);
            c.floats("continuation legacy mixer",continuation.mixer,nb.mixer_output,gdn_gate);
            reference.commit(2);parallel.commit(2);committed(c,reference.snapshot(),parallel.snapshot(),continuation);++cases;
            if(rccl){hybrid_parallel->commit(2);exact_hybrid_layers(c,*hybrid_parallel,parallel);++rccl_cases;}
        }
        if(profiled){
            // Instrumentation may change scheduling. It never replaces the
            // authoritative unprofiled all-prefix gate immediately above.
            parallel.set_execution(execution,true);
            for(int tokens=1;tokens<=8;++tokens){
                std::printf("PROFILE_DIAGNOSTIC mode=%s T=%d separate-from-correctness-and-timing\n",execution_name(execution),tokens);
                const auto input=signed_input(size_t(tokens)*H*N,43+tokens*19,0.65f);
                reference.reset_state(state,conv);parallel.reset_state(state,conv);
                reference.propose(input,tokens,++epoch);parallel.propose(input,tokens,epoch);
                profile_report(c,parallel);const auto a=reference.snapshot(),b=parallel.snapshot();seams(c,a,b,false,true,association,&full.weights(),hybrid);seams(c,b,parallel.snapshot(1),true);
                reference.commit(0);parallel.commit(0);
            }
            parallel.set_execution(execution,false);
        }
        }
        c.require(routes_changed,"signed fixtures exercise changing route IDs");
        c.require(repeated_ids,"multi-token fixtures exercise repeated expert IDs");
        std::printf("\n%s whole-GDN engineering parity cases=%d continuation-cases=%d failures=%d; no full-model inference/quality/performance claim\n",c.failures?"FAIL":"PASS",cases,cases,c.failures);
        if(c.failures)return 1;
        if(use_rccl){
            if(rccl_cases!=44)throw std::runtime_error("RCCL all-prefix gate incomplete; timing prohibited");
            std::printf("RCCL_LAYER_EXACT_GATE_PASS prefix_cases=%d continuation_cases=%d T=1..8 orders=2 banks=2 old_hybrid_tolerance_bits=0 original_full_gates=unchanged\n",rccl_cases,rccl_cases);
            rccl_parallel->set_rccl_launch_for_test(o.rccl_first);
        }
        if(o.calibrate)calibration(c,o,g,reference,row_parallel,full.weights(),epoch,state,conv);
        if(o.benchmark){
            std::vector<BenchStudy> studies;
            if(use_rccl){
                studies.push_back({reference,*rccl_parallel,tp::TpGdnExecution::Captured,tp::TpGdnExecution::HybridRcclCaptured,"single-gpu-captured","tp-hybrid-rccl"});
                studies.push_back({*hybrid_parallel,*rccl_parallel,tp::TpGdnExecution::HybridCaptured,tp::TpGdnExecution::HybridRcclCaptured,"tp-hybrid-captured","tp-hybrid-rccl",true});
            }else if(o.execution=="hybrid"){
                studies.push_back({reference,*hybrid_parallel,tp::TpGdnExecution::Captured,tp::TpGdnExecution::HybridCaptured,"single-gpu-captured","tp-hybrid-captured"});
                studies.push_back({row_parallel,*hybrid_parallel,tp::TpGdnExecution::Captured,tp::TpGdnExecution::HybridCaptured,"tp-row-captured","tp-hybrid-captured"});
            }else if(!o.execution_explicit){
                studies.push_back({reference,*column_parallel,tp::TpGdnExecution::ColumnCaptured,tp::TpGdnExecution::ColumnCaptured,"single-gpu-captured","tp-column-captured"});
                studies.push_back({row_parallel,*column_parallel,tp::TpGdnExecution::Captured,tp::TpGdnExecution::ColumnCaptured,"tp-row-captured","tp-column-captured"});
            }else for(auto mode:executions)studies.push_back({reference,mode==tp::TpGdnExecution::ColumnCaptured?*column_parallel:row_parallel,mode,mode,"single-gpu",mode==tp::TpGdnExecution::ColumnCaptured?"tp-column-captured":"tp-output-rows"});
            for(const auto& study:studies){benchmark(c,o,study,full.weights(),epoch,state,conv);sustained_benchmark(c,o,study,full.weights(),epoch,state,conv);}
            std::printf("BENCH_GATE PASS burst-pairs-every-result=checked block-shadow-every-step=checked measured-block-final-only=checked; no measured block intermediate downloads; one-layer wall times only\n");}
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"tp2_gdn_layer: %s\n",e.what());return 2;}
}
