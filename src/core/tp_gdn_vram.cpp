#include "tp_gdn_vram.hpp"
#include "tp_gdn_rccl_workers.hpp"

#include <hip/hip_runtime.h>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

namespace strata::core::tp2::detail {
namespace {
void hip_check(hipError_t result, const char* what) {
    if(result!=hipSuccess)
        throw std::runtime_error(std::string("TP GDN VRAM ")+what+": "+hipGetErrorString(result));
}
[[noreturn]] void unsafe_teardown(const char* reason) noexcept {
    std::fprintf(stderr,"VRAM_LAYER_FAIL reason=%s exit=4 timing_admitted=0 "
                        "teardown=process_exit_without_buffer_release\n",reason);
    std::fflush(nullptr);
    std::_Exit(4);
}
}

struct VramSession::Impl {
    std::array<int,2> devices;
    std::array<void*,2> compute,copy;
    int watchdog_ms;
    bool closed=false;
    RcclWorkers workers{"tp_gdn_vram"};

    Impl(const std::array<int,2>& d,const std::array<void*,2>& c,
         const std::array<void*,2>& x,int timeout):devices(d),compute(c),copy(x),watchdog_ms(timeout) {
        workers.run("initialize-owners",[&](int rank){
            hip_check(hipSetDevice(devices[rank]),"select owner device");
            if(std::fesetround(FE_TONEAREST)!=0)
                throw std::runtime_error("TP GDN VRAM cannot set worker rounding");
            hipDeviceProp_t properties{};
            hip_check(hipGetDeviceProperties(&properties,devices[rank]),"device properties");
            const std::string arch(properties.gcnArchName);
            if(arch!="gfx906"&&arch.rfind("gfx906:",0)!=0)
                throw std::runtime_error("TP GDN VRAM requires two gfx906 devices");
        },watchdog_ms);
    }
    void run(const char* stage,const std::function<void(int)>& action) {
        if(closed)throw std::logic_error("TP GDN VRAM session already closed");
        workers.run(stage,action,watchdog_ms);
    }
    void drain() noexcept {
        if(closed)return;
        try {
            run("drain-both",[&](int rank){
                hip_check(hipSetDevice(devices[rank]),"drain device");
                hip_check(hipStreamSynchronize(static_cast<hipStream_t>(compute[rank])),"drain compute");
                hip_check(hipStreamSynchronize(static_cast<hipStream_t>(copy[rank])),"drain copy");
            });
        }catch(const std::exception& e){unsafe_teardown(e.what());}
        catch(...){unsafe_teardown("unknown drain failure");}
    }
};

VramSession::VramSession(const std::array<int,2>& devices,
                         const std::array<void*,2>& compute,
                         const std::array<void*,2>& copy,int watchdog_ms) {
    if(devices[0]<0||devices[1]<0||devices[0]==devices[1]||
       !compute[0]||!compute[1]||!copy[0]||!copy[1]||
       watchdog_ms<100||watchdog_ms>120000)
        throw std::invalid_argument("TP GDN VRAM invalid rank streams or watchdog");
    impl_=std::make_unique<Impl>(devices,compute,copy,watchdog_ms);
}
VramSession::~VramSession() {
    if(impl_&&!impl_->closed)unsafe_teardown("session destroyed before paired shutdown");
}
void VramSession::paired(const char* stage,const std::function<void(int)>& action) {impl_->run(stage,action);}
void VramSession::check_peer() const {impl_->workers.check_peer();}
void VramSession::complete(int rank) {
    if(!impl_->workers.owns_rank(rank))throw std::logic_error("TP GDN VRAM wrong stream owner");
    for(;;){
        impl_->workers.check_peer();
        const auto status=hipStreamQuery(static_cast<hipStream_t>(impl_->compute[rank]));
        if(status==hipSuccess)return;
        if(status!=hipErrorNotReady)hip_check(status,"query completion");
        std::this_thread::yield();
    }
}
void VramSession::drain() noexcept {impl_->drain();}
void VramSession::shutdown(const std::function<void(int)>& destroy_graphs,
                           const std::function<void(int)>& release_rank_resources) noexcept {
    if(impl_->closed)return;
    impl_->drain();
    try {
        // Both drains finish before either graph or peer-addressable allocation
        // is released. Each later API remains covered by the worker watchdog.
        impl_->run("shutdown-graphs",destroy_graphs);
        impl_->run("shutdown-rank-resources",release_rank_resources);
        impl_->closed=true;
    }catch(const std::exception& e){unsafe_teardown(e.what());}
    catch(...){unsafe_teardown("unknown shutdown failure");}
}
} // namespace strata::core::tp2::detail
