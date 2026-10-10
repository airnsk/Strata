// CPU-only call-option and scratch-offset contract; no GPU dispatch validation.
#include "strata/kernels/iq_kernels.hpp"
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
using namespace strata::kernels;
static void need(bool ok) { if (!ok) throw std::runtime_error("native expert call policy mismatch"); }
template<class F> static void rejects(F f) { bool bad=false; try { f(); } catch(const std::exception&) { bad=true; } need(bad); }
int main() {
  try {
    for (auto phase : {NativeExpertPhase::Full, NativeExpertPhase::GateUp, NativeExpertPhase::Down}) {
      for (int mode : {-1,0,1,2,4,5,6,7,8}) need(native_expert_call_options_valid({phase,mode}));
      for (int mode : {-2,3,9,999}) need(!native_expert_call_options_valid({phase,mode}));
    }
    need(!native_expert_call_options_valid({static_cast<NativeExpertPhase>(-1),8}));
    need(!native_expert_call_options_valid({static_cast<NativeExpertPhase>(3),8}));
    for (int entries : {0,1,10,80,512}) for (int width : {32,320,640,2560}) {
      const std::size_t plane=static_cast<std::size_t>(entries)*width*4;
      const std::size_t expected=3*((plane+255)/256)*256;
      need(native_expert_hidden_q8_offset(entries,width)==expected);
    }
    rejects([] { (void)native_expert_hidden_q8_offset(-1,640); });
    rejects([] { (void)native_expert_hidden_q8_offset(1,0); });
    rejects([] { (void)native_expert_hidden_q8_offset(1,319); });
    rejects([] { (void)native_expert_hidden_q8_offset(std::numeric_limits<int64_t>::max(),640); });
    rejects([] { (void)native_expert_hidden_q8_offset(1,std::numeric_limits<int64_t>::max()-31); });
    std::puts("PASS: native expert explicit options and hidden-Q8 scratch offsets (CPU only)");
    return 0;
  } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what());return 1; }
}
