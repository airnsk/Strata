// Focused real-weight HC subsystem gate. No model/TP/generation speed claim.
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace k=strata::kernels;
namespace {
    constexpr int N=2560,H=4,D=N*H,L=320;
    void ck(cudaError_t e){
        if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    }
    struct Buffer {
        void* p=nullptr;
        size_t bytes;
        explicit Buffer(size_t n):bytes(n){
            ck(cudaMalloc(&p,n));
            ck(cudaMemset(p,0,n));
        }
        ~Buffer(){
            cudaFree(p);
        }
        Buffer(const Buffer&)=delete;
        Buffer&operator=(const Buffer&)=delete;
        float* f(){
            return static_cast<float*>(p);
        }
        std::vector<unsigned char> get()const{
            std::vector<unsigned char> v(bytes);
            ck(cudaMemcpy(v.data(),p,bytes,cudaMemcpyDeviceToHost));
            return v;
        }
    };
    uint64_t hash(const void* p,size_t n){
        uint64_t h=14695981039346656037ull;
        const auto* b=static_cast<const unsigned char*>(p);
        for(size_t i=0;i<n;++i){
            h^=b[i];
            h*=1099511628211ull;
        }
        return h;
    }
    struct Options {
        std::string pack;
        std::vector<std::string> shards;
        int trials=8,batch=32,warmup=16;
    };
    Options parse(int argc,char**argv){
        Options o;
        for(int i=1;i<argc;++i){
            std::string a=argv[i];
            if(i+1==argc)throw std::runtime_error("missing option value");
            std::string v=argv[++i];
            if(a=="--pack")o.pack=v;
            else if(a=="--gguf")o.shards.push_back(v);
            else {
                size_t used=0;
                int n=std::stoi(v,&used);
                if(used!=v.size())throw std::runtime_error("invalid integer");
                if(a=="--trials")o.trials=n;
                else if(a=="--batch")o.batch=n;
                else if(a=="--warmup")o.warmup=n;
                else throw std::runtime_error("unknown option "+a);
            }
        }
        if(o.pack.empty()||o.shards.empty()||o.trials<2||o.trials>100||o.trials%2||o.batch<1||o.batch>1024||o.warmup<1||o.warmup>1024)throw std::runtime_error("usage: hc_retiled_bench --pack PACK --gguf SHARD [--gguf SHARD] [--trials 8] [--batch 32] [--warmup 16]");
        return o;
    }
    struct Weights {
        strata::core::WeightTable table;
        Buffer arena;
        static uint64_t size(const Options& o,std::set<std::string>& skip){
            std::set<std::string> keep,seen;
            for(const char* side:{"attn","ffn"})for(const char* role:{"norm","down","up","inject"})keep.insert(std::string("blk.0.hc_")+side+"_"+role+".weight");
            std::ifstream in(o.pack+"/index.txt");
            if(!in)throw std::runtime_error("missing pack index");
            std::string line,name;
            bool aligned=false;
            while(std::getline(in,line)){
                std::istringstream s(line);
                if(!(s>>name))throw std::runtime_error("empty index row");
                if(name[0]=='#'){
                    std::string label;
                    int a;
                    if(name=="#"&&(s>>label)&&label=="align"){
                        if(!(s>>a)||a<=0||a>4096||(a&(a-1)))throw std::runtime_error("bad index alignment");
                        aligned=true;
                    }
                    continue;
                }
                if(line.size()>=1000||name.size()>255||!seen.insert(name).second)throw std::runtime_error("duplicate/oversized index row");
                std::array<int64_t,18> f{};
                for(auto& x:f)if(!(s>>x))throw std::runtime_error("malformed index row");
                std::string extra;
                if(s>>extra)throw std::runtime_error("extra index field");
                if(f[6]<=0||f[7]<0||f[6]>INT64_MAX/std::max<int64_t>(f[7],1))throw std::runtime_error("invalid dimensions");
                if(!keep.count(name)){
                    skip.insert(name);
                    continue;
                }
                const bool norm=name.ends_with("norm.weight");
                const int64_t count=f[6]*std::max<int64_t>(f[7],1);
                if(count>(64ll<<20)/4||(norm?f[1]!=2:(f[1]!=1&&f[1]!=4))||f[5]!=count*(norm?4:2)||f[3]!=count*(f[1]==4?2:4)||f[2]<0||f[4]<0||f[8]||f[11]||f[12]||f[13]||f[14]||f[15]||f[16]||(f[0]!=0&&f[0]!=3))throw std::runtime_error("unsupported canonical HC representation");
                auto bytes=std::filesystem::file_size(o.pack+(f[0]==0?"/dense.bin":"/extra.bin"));
                if(uint64_t(f[2])>bytes||uint64_t(f[3])>bytes-uint64_t(f[2]))throw std::runtime_error("HC source span out of bounds");
            }
            if(!aligned)throw std::runtime_error("missing index alignment");
            for(const auto& n:keep)if(!seen.count(n))throw std::runtime_error("missing HC tensor "+n);
            uint64_t bytes=0;
            std::string err;
            if(!strata::core::WeightTable::pool_bytes(o.pack,bytes,err,&skip))throw std::runtime_error(err);
            if(!bytes||bytes>(64ull<<20))throw std::runtime_error("HC arena size invalid");
            return bytes;
        }
        Weights(const Options&o,std::set<std::string>&skip):arena(size(o,skip)){
            std::string err;
            if(!table.load(o.pack,arena.p,arena.bytes,err,&skip))throw std::runtime_error(err);
            strata::GgufModel model(o.shards);
            for(int side=0;side<2;++side)for(int role=0;role<4;++role){
                const auto& w=get(side,role);
                const auto name=tensor(side,role);
                const auto* src=model.find(name);
                if(!src||src->type!=uint32_t(role?30:0)||src->shape.empty()||src->shape.size()>2||src->shape[0]!=uint64_t(w.ne0)||(src->shape.size()==2?src->shape[1]:1)!=uint64_t(std::max<int64_t>(w.ne1,1)))throw std::runtime_error("GGUF HC shape mismatch");
                std::vector<unsigned char> data(w.bytes);
                ck(cudaMemcpy(data.data(),w.data,w.bytes,cudaMemcpyDeviceToHost));
                std::printf("WEIGHT name=%s source-type=%s source-type-id=%u runtime=%s bytes=%llu fnv1a64=%016llx\n",name.c_str(),strata::ggml_type_name(src->type),src->type,role?"BF16":"F32",(unsigned long long)w.bytes,(unsigned long long)hash(data.data(),data.size()));
            }
            ck(cudaDeviceSynchronize());
        }
        static std::string tensor(int side,int role){
            const char* names[]={"norm","down","up","inject"};
            return std::string("blk.0.hc_")+(side?"ffn_":"attn_")+names[role]+".weight";
        }
        const strata::core::WeightRef& get(int side,int role)const{
            const auto* w=table.find(tensor(side,role));
            const int widths[]={D,D,L,D},counts[]={1,L,D,H};
            if(!w||!w->resident||!w->data||w->quantized()||w->ne0!=widths[role]||std::max<int64_t>(w->ne1,1)!=counts[role]||w->elements!=int64_t(widths[role])*counts[role]||w->bytes!=uint64_t(widths[role])*counts[role]*(role?2:4)||w->kind!=(role?strata::core::WeightKind::Bf16InF32:strata::core::WeightKind::F32))throw std::runtime_error("invalid runtime HC tensor");
            return *w;
        }
    };
    struct Stream {
        cudaStream_t p{};
        Stream(){
            ck(cudaStreamCreateWithFlags(&p,cudaStreamNonBlocking));
        }
        ~Stream(){
            cudaStreamDestroy(p);
        }
    };
    struct Graph {
        cudaGraph_t g{};
        cudaGraphExec_t e{};
        ~Graph(){
            if(e)cudaGraphExecDestroy(e);
            if(g)cudaGraphDestroy(g);
        }
    };
    struct Work {
        int t;
        Buffer input,block0,block1,residual,lo,rs,inject,mixed,xn;
        std::array<k::FusedGrArgs,8> args[2];
        explicit Work(int n):t(n),input(size_t(n)*D*4),block0(size_t(n)*N*4),block1(size_t(n)*N*4),residual(size_t(n)*D*4),lo(size_t(2*n)*L*4),rs(size_t(2*n)*H*4),inject(size_t(2*n)*H*4),mixed(size_t(2*n)*N*4),xn(size_t(2*n)*D*4){}
        void fixture(uint32_t seed,float scale){
            for(Buffer* b:{&input,&block0,&block1}){
                std::vector<float> v(b->bytes/4);
                for(auto& x:v){
                    seed=1664525u*seed+1013904223u;
                    x=(int(seed>>8)-8388608)*(scale/8388608.f);
                }
                ck(cudaMemcpy(b->p,v.data(),b->bytes,cudaMemcpyHostToDevice));
            }
            ck(cudaDeviceSynchronize());
        }
        void launch(const Weights&w,bool candidate,cudaStream_t s,unsigned long long* stamps=nullptr){
            for(int side=0;side<2;++side){
                if(stamps)k::gpu_stamp(stamps,side*4,s);
                for(int i=0;i<t;++i){
                    auto&a=args[side][i];
                    a.R=input.f()+i*D;
                    a.R_out=residual.f()+i*D;
                    a.apply=side;
                    a.bo_prev=block0.f()+i*N;
                    a.inj_prev=inject.f()+i*H;
                    a.w_norm=static_cast<const float*>(w.get(side,0).data);
                    a.w_down=static_cast<const uint16_t*>(w.get(side,1).data);
                    a.w_up=static_cast<const uint16_t*>(w.get(side,2).data);
                    a.w_inject=static_cast<const uint16_t*>(w.get(side,3).data);
                    a.lo=lo.f()+(side*t+i)*L;
                    a.rs=rs.f()+(side*t+i)*H;
                    a.inject_out=inject.f()+(side*t+i)*H;
                    a.mixed=mixed.f()+(side*t+i)*N;
                }
                if(candidate)k::fused_gr_read_multi_retiled(args[side].data(),t,xn.f()+side*t*D,s,stamps,side*4+1);
                else k::fused_gr_read_multi(args[side].data(),t,xn.f()+side*t*D,s,stamps,side*4+1);
                if(stamps)k::gpu_stamp(stamps,side*4+3,s);
            }
            k::gr_write_multi(residual.f(),block1.f(),inject.f()+t*H,{N,H,L}
            ,residual.f(),t,s);
            if(stamps)k::gpu_stamp(stamps,8,s);
        }
        std::vector<std::vector<unsigned char>> snapshot(){
            std::vector<std::vector<unsigned char>> out;
            for(Buffer*b:{&lo,&rs,&inject,&mixed,&xn,&residual})out.push_back(b->get());
            return out;
        }
    };
    void equal(const std::vector<std::vector<unsigned char>>&a,const std::vector<std::vector<unsigned char>>&b){
        const char* names[]={"lo","rs","inject","mixed","xn","residual"};
        for(size_t j=0;j<a.size();++j){
            if(a[j].size()!=b[j].size())throw std::runtime_error("snapshot size");
            size_t bad=0,nonfinite=0;
            for(size_t i=0;i<a[j].size();i+=4){
                float x,y;
                std::memcpy(&x,a[j].data()+i,4);
                std::memcpy(&y,b[j].data()+i,4);
                nonfinite+=!std::isfinite(x)||!std::isfinite(y);
                bad+=std::memcmp(a[j].data()+i,b[j].data()+i,4)!=0;
            }
            if(bad||nonfinite){
                std::printf("EXACT_FAIL stage=%s differing-floats=%zu nonfinite=%zu\n",names[j],bad,nonfinite);
                throw std::runtime_error("HC exact gate failed; timing invalid");
            }
        }
    }
    struct Sample {
        double wall,gpu;
    };
    Sample run(Graph&g,Stream&s,int count){
        cudaEvent_t a{}
        ,b{};
        ck(cudaEventCreateWithFlags(&a,0));
        ck(cudaEventCreateWithFlags(&b,0));
        auto start=std::chrono::steady_clock::now();
        ck(cudaEventRecord(a,s.p));
        for(int i=0;i<count;++i)ck(cudaGraphLaunch(g.e,s.p));
        ck(cudaEventRecord(b,s.p));
        ck(cudaStreamSynchronize(s.p));
        auto stop=std::chrono::steady_clock::now();
        float ms=0;
        ck(cudaEventElapsedTime(&ms,a,b));
        if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("invalid GPU timing");
        ck(cudaEventDestroy(a));
        ck(cudaEventDestroy(b));
        return {std::chrono::duration<double,std::micro>(stop-start).count()/count,ms*1000./count};
    }
    double median(std::vector<double> v){
        std::sort(v.begin(),v.end());
        return (v[(v.size()-1)/2]+v[v.size()/2])/2;
    }
}
int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    try{
        auto o=parse(argc,argv);
        for(const char* key:{"STRATA_HC_PERSIST","STRATA_GR_V3","STRATA_GR_SPLIT","STRATA_HC_Q8","STRATA_QFUSE","STRATA_NO_MULTI_GR","STRATA_GR_DOWN_MAX4","STRATA_TSUM"}){
            const char*v=std::getenv(key);
            if(v&&std::strcmp(v,"0"))throw std::runtime_error(std::string("requires disabled ")+key);
            setenv(key,"0",1);
        }
        if(std::getenv("STRATA_HC_SPLIT")||std::getenv("STRATA_GR_FAST"))throw std::runtime_error("unset STRATA_HC_SPLIT and STRATA_GR_FAST: benchmark selects accepted default");
        k::gr_set_native_mmvf(true);
        k::fused_gr_set_persistent(0);
        k::fused_gr_set_fast(-1);
        std::printf("HC_GATE exact-finite-float-bits all-intermediates-and-residual; arithmetic-expectation=identical-reduction-order; scope=two-real-layer0-HC-reads-plus-final-write synthetic-mixer-outputs; no-model-quality-or-generation-claim\n");
        std::printf("HC_TIMING unprofiled-graphs whole-HC wall-and-GPU-us/per-call batch=%d warmup=%d crossovers=%d same-input-ABBA randomized-first-order seed=20261010 setup-downloads-excluded measured-batch-final-only-check; repeated-calls-have-identical-input\n",o.batch,o.warmup,o.trials);
        int count=0;
        ck(cudaGetDeviceCount(&count));
        if(count<2)throw std::runtime_error("requires two GPUs for sequential per-device comparison");
        for(int device=0;device<2;++device){
            ck(cudaSetDevice(device));
            cudaDeviceProp prop{};
            ck(cudaGetDeviceProperties(&prop,device));
            std::printf("DEVICE ordinal=%d name=%s pci=%04x:%02x:%02x\n",device,prop.name,prop.pciDomainID,prop.pciBusID,prop.pciDeviceID);
#if defined(STRATA_HIP_GFX906)
            int wall_clock_khz=0;
            ck(hipDeviceGetAttribute(&wall_clock_khz,hipDeviceAttributeWallClockRate,device));
            if(wall_clock_khz!=25000)throw std::runtime_error("GPU stamp conversion requires 25000kHz wall clock");
            std::printf("STAMP_CLOCK device=%d wall-clock-khz=%d ns-per-tick=40\n",device,wall_clock_khz);
#endif
            k::fused_gr_check();
            std::printf("BASELINE device=%d accepted-variant=%d fast=default persistent=0 v3=0 split-override=0 q8=0 qfuse=0\n",device,k::fused_gr_variant());
            std::set<std::string> skip;
            Weights weights(o,skip);
            Stream stream;
            for(int t=1;t<=8;++t){
                k::FusedGrRetileInfo info{};
                k::fused_gr_retiled_info(t,&info);
                std::printf("TILING device=%d T=%d selected-variant=%d tile-floats=%d chunk-tokens=%d candidate-rows=%d candidate-threads=%d candidate-blocks=%d candidate-lds=%llu original-rows=8 original-threads=256 original-blocks=41\n",device,t,info.variant,info.tile_floats,info.chunk_tokens,info.rows_per_block,info.threads,info.blocks,(unsigned long long)info.dynamic_lds_bytes);
                Work work(t);
                Graph graphs[2];
                work.fixture(20261010u+t,1.f);
                for(int arm=0;arm<2;++arm){
                    work.launch(weights,arm,stream.p);
                    ck(cudaStreamSynchronize(stream.p));
                    ck(cudaStreamBeginCapture(stream.p,cudaStreamCaptureModeThreadLocal));
                    work.launch(weights,arm,stream.p);
                    ck(cudaStreamEndCapture(stream.p,&graphs[arm].g));
                    ck(cudaGraphInstantiate(&graphs[arm].e,graphs[arm].g,nullptr,nullptr,0));
                }
                // Authoritative unprofiled whole-chain parity for every T, before diagnostics.
                run(graphs[0],stream,1);
                auto chain_reference=work.snapshot();
                run(graphs[1],stream,1);
                equal(chain_reference,work.snapshot());
                // Extra direct-launch correctness covers final mixers, aliasing and every T.
                for(int apply=0;apply<2;++apply)for(int null_inject=0;null_inject<2;++null_inject)for(int inplace=0;inplace<2;++inplace){
                    std::vector<std::vector<unsigned char>> ref;
                    for(int arm=0;arm<2;++arm){
                        work.fixture(701u+uint32_t(t),1.f);
                        for(Buffer*b:{&work.lo,&work.rs,&work.inject,&work.mixed,&work.xn,&work.residual}){
                            ck(cudaMemset(b->p,0,b->bytes));
                        }
                        ck(cudaDeviceSynchronize());
                        // First construct normal arguments, then isolate one read.
                        work.launch(weights,false,stream.p);
                        ck(cudaStreamSynchronize(stream.p));
                        work.fixture(701u+uint32_t(t),1.f);
                        for(Buffer*b:{&work.lo,&work.rs,&work.inject,&work.mixed,&work.xn,&work.residual}){
                            ck(cudaMemset(b->p,0,b->bytes));
                        }
                        ck(cudaDeviceSynchronize());
                        auto a=work.args[0];
                        for(int i=0;i<t;++i){
                            a[i].apply=apply;
                            a[i].R_out=inplace?work.input.f()+i*D:work.residual.f()+i*D;
                            a[i].inj_prev=work.block1.f()+i*N;
                            a[i].w_inject=null_inject?nullptr:a[i].w_inject;
                        }
                        if(arm)k::fused_gr_read_multi_retiled(a.data(),t,work.xn.f(),stream.p);
                        else k::fused_gr_read_multi(a.data(),t,work.xn.f(),stream.p);
                        ck(cudaStreamSynchronize(stream.p));
                        auto got=work.snapshot();
                        got.push_back(work.input.get());
                        // Snapshot helper's named regions plus input (alias writeback).
                        if(arm){
                            auto lhs=ref.back(),rhs=got.back();
                            ref.pop_back();
                            got.pop_back();
                            equal(ref,got);
                            if(lhs!=rhs)throw std::runtime_error("in-place residual exact mismatch");
                            for(size_t j=0;j<lhs.size();j+=4){
                                float v;
                                std::memcpy(&v,lhs.data()+j,4);
                                if(!std::isfinite(v))throw std::runtime_error("nonfinite input/writeback");
                            }
                        } else ref=got;
                    }
                }
                std::printf("HC_EXACT device=%d T=%d apply=both inject=present/null output=alias/separate PASS\n",device,t);
                // Separate diagnostic graph only: timestamps alter execution scheduling.
                work.fixture(20261010u+t,1.f);
                Buffer stamps(9*sizeof(unsigned long long));
                std::vector<std::vector<unsigned char>> diagnostic_ref;
                for(int arm=0;arm<2;++arm){
                    Graph diagnostic;
                    ck(cudaStreamBeginCapture(stream.p,cudaStreamCaptureModeThreadLocal));
                    work.launch(weights,arm,stream.p,static_cast<unsigned long long*>(stamps.p));
                    ck(cudaStreamEndCapture(stream.p,&diagnostic.g));
                    ck(cudaGraphInstantiate(&diagnostic.e,diagnostic.g,nullptr,nullptr,0));
                    run(diagnostic,stream,o.warmup);
                    auto result=work.snapshot();
                    if(arm)equal(diagnostic_ref,result);
                    else diagnostic_ref=result;
                    auto raw=stamps.get();
                    std::array<unsigned long long,9> ts{};
                    std::memcpy(ts.data(),raw.data(),raw.size());
                    for(size_t i=1;i<ts.size();++i)if(ts[i]<ts[i-1])throw std::runtime_error("nonmonotonic GPU stamps");
                    std::printf("HC_DIAGNOSTIC device=%d T=%d arm=%s attn-norm-ns=%llu attn-down-ns=%llu attn-up-ns=%llu ffn-norm-ns=%llu ffn-down-ns=%llu ffn-up-ns=%llu final-write-ns=%llu instrumented-only=1\n",device,t,arm?"candidate":"original",ts[1]-ts[0],ts[2]-ts[1],ts[3]-ts[2],ts[5]-ts[4],ts[6]-ts[5],ts[7]-ts[6],ts[8]-ts[7]);
                }
                if(t==3||t==6||t==7)continue;
                std::vector<double> wall[2],gpu[2],ratios;
                for(int trial=0;trial<o.trials;++trial){
                    const float scales[]={0.03125f,1.f,8.f};
                    work.fixture(20261010u+uint32_t(t*1000+trial),scales[trial%3]);
                    auto input=work.input.get();
                    std::printf("FIXTURE device=%d T=%d trial=%d scale=%g input-fnv1a64=%016llx\n",device,t,trial,scales[trial%3],(unsigned long long)hash(input.data(),input.size()));
                    for(int arm=0;arm<2;++arm)run(graphs[arm],stream,o.warmup);
                    run(graphs[0],stream,1);
                    auto expected=work.snapshot();
                    run(graphs[1],stream,1);
                    equal(expected,work.snapshot());
                    Sample samples[4];
                    const int first=int((uint32_t(trial/2)*1664525u+20261010u)>>7)&1;
                    const int order=(trial%2)?1-first:first;
                    const int arms[]={order,1-order,1-order,order};
                    for(int leg=0;leg<4;++leg){
                        int arm=arms[leg];
                        run(graphs[arm],stream,o.warmup);
                        samples[leg]=run(graphs[arm],stream,o.batch);
                        auto actual=work.snapshot();
                        equal(expected,actual);
                        wall[arm].push_back(samples[leg].wall);
                        gpu[arm].push_back(samples[leg].gpu);
                        std::printf("HC_RAW device=%d T=%d trial=%d leg=%d arm=%s wall-us=%.6f gpu-us=%.6f exact=PASS\n",device,t,trial,leg,arm?"candidate":"original",samples[leg].wall,samples[leg].gpu);
                    }
                    double means[2]={};
                    for(int leg=0;leg<4;++leg)means[arms[leg]]+=samples[leg].wall/2;
                    ratios.push_back(means[0]/means[1]);
                    std::printf("HC_CROSSOVER device=%d T=%d trial=%d original-us=%.6f candidate-us=%.6f ratio=%.6f\n",device,t,trial,means[0],means[1],ratios.back());
                }
                std::printf("HC_SUMMARY device=%d T=%d original-wall-median-us=%.6f candidate-wall-median-us=%.6f original-gpu-median-us=%.6f candidate-gpu-median-us=%.6f crossed-wall-ratio-median=%.6f exact=PASS\n",device,t,median(wall[0]),median(wall[1]),median(gpu[0]),median(gpu[1]),median(ratios));
            }
        }
        std::puts("HC_RETILED_GATE PASS exact correctness and whole-HC timing complete on both devices");
        return 0;
    }
    catch(const std::exception&e){
        std::fprintf(stderr,"HC_RETILED_GATE FAIL %s\n",e.what());
        return 1;
    }
}
