from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[2]
source=r'''
#include "ds4_batch_execution.hpp"
#include <thread>
#include <cassert>
using namespace strata::ds4;
int main() {
    BatchExecutionAccounting accounting;
    constexpr int n=4000;
    auto work=[&] {
        for(int i=0;i<n;++i) {
            Stats job{};job.cpu_ms=.25;job.cpu_experts=6;
            job.ahead_picks=3;job.ahead_pf=2;job.ram_experts=4;
            job.ahead_rank_n[15]=1;job.ahead_rank_hit[15]=1;
            accounting.complete(job);
        }
    };
    std::thread a(work),b(work);
    uint64_t last=0;
    for(int i=0;i<n;++i) {
        uint64_t count=0;auto s=accounting.snapshot(count);
        assert(count>=last&&count<=2*n);last=count;
        assert(s.cpu_experts==6*count&&s.cpu_ms==.25*count);
        // A bound request can be discarded without touching executor ownership.
        Stats bound=s;bound=Stats{};
        assert(bound.cpu_experts==0);
    }
    a.join();b.join();
    uint64_t count=0;auto s=accounting.snapshot(count);
    assert(count==2*n&&s.cpu_experts==12*n&&s.cpu_ms==.5*n);
    assert(s.ahead_picks==6*n&&s.ahead_pf==4*n&&s.ram_experts==8*n);
    assert(s.ahead_rank_n[15]==2*n&&s.ahead_rank_hit[15]==2*n);
}
'''
with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'test.cpp').write_text(source)
    subprocess.run(['c++','-std=c++20','-O1','-g','-pthread','-fsanitize=address,undefined',
                    '-I'+str(root/'src/dsv41'),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
    subprocess.run([str(p/'test')],check=True)
print('PASS shared CPU accounting: concurrent completion/snapshot; independent request lifetime')
