"""Ownership/lifecycle checks only; this does not claim resident GPU batching."""
from pathlib import Path
import subprocess, tempfile
root=Path(__file__).resolve().parents[2]
source=r'''
#include "ds4_request_state.hpp"
#include "ds4_request_slots.hpp"
#include "ds4_yoco_pipeline.hpp"
#include <cassert>
#include <limits>
#include <utility>
using namespace strata::ds4;
struct Probability { int n=0; };
int main() {
    assert(request_saved_position_fits(128,128,128));
    assert(!request_saved_position_fits(128,129,129));
    assert(!request_saved_position_fits(128,0,0));
    assert(!request_saved_position_fits(128,-1,UINT64_MAX));
    assert(!request_saved_position_fits(128,127,128));
    assert(request_saved_position_fits(262144,262144,262144));
    using State=RequestHostState<Probability,16>;
    State a,b;
    a.pos=129;a.toks={7,8};a.ctoks={70,80};a.ck[0].pos=128;
    a.ck[0].win={{1,2}};a.ck[0].comp={{3.0f}};
    a.cur_topk_n[0]=7;a.cur_cand_n[0]=9;a.n_logits=1;
    a.dq_seed=11;a.dq_counter=29;a.h_specq[0].n=5;a.drafts_q=true;
    a.st.tokens=129;a.logits_all=true;a.chunk_logits={3.5f};a.halted=true;
    a.cancel=[](){return true;};
    a.imgs.push_back(Image{3,std::make_shared<const std::vector<uint16_t>>(4,19)});
    b.pos=256;b.toks={99};b.dq_seed=22;b.dq_counter=3;
    std::swap(a,b);
    assert(a.pos==256&&a.toks[0]==99&&a.dq_seed==22&&a.dq_counter==3);
    assert(b.pos==129&&b.ck[0].pos==128&&b.ck[0].win[0][1]==2);
    assert(b.cur_topk_n[0]==7&&b.cur_cand_n[0]==9&&b.h_specq[0].n==5);
    assert(b.imgs[0].start==3&&b.st.tokens==129&&b.drafts_q);
    assert(b.logits_all&&b.chunk_logits[0]==3.5f&&b.halted&&b.cancel());
    assert(!a.cancel&&!a.logits_all);
    a=State{};
    assert(a.pos==0&&a.toks.empty()&&a.ck[0].pos==-1&&a.imgs.empty());
    assert(b.toks[0]==7&&b.ck[0].comp[0][0]==3.0f);
    // Non-owning views bind pointers; swapping views never copies model weights.
    uint8_t ringA[128]={1},ringB[128]={2},kvA[3]={3},kvB[3]={4};
    LayerRequestViews la,lb;la.win=ringA;lb.win=ringB;std::swap(la,lb);
    assert(la.win==ringB&&lb.win==ringA);
    SourceRequestViews sa,sb;sa.cap=128;sb.cap=256;
    sa.ckv={kvA};sb.ckv={kvB};std::swap(sa,sb);
    assert(sa.cap==256&&sa.ckv[0]==kvB&&sb.ckv[0]==kvA);
    RequestSlots<4> slots;
    RequestLease ids[4];
    for(auto& id:ids){auto x=slots.admit(129,256,6,512);assert(x);id=*x;}
    assert(!slots.admit(1,1,0,8)); // fifth is bounded, not silently recycled
    auto w=slots.begin(ids[0]);auto handoff=slots.begin(ids[0]);
    assert(w&&handoff&&slots.can_deliver(*w));
    assert(!slots.decode_ready(ids[0]));
    assert(slots.finish(ids[0],true));
    assert(!slots.can_deliver(*w)&&!slots.release(ids[0]));
    assert(slots.acknowledge(*w)&&!slots.acknowledge(*w));
    assert(!slots.terminal_ready(ids[0])); // handoff still owns memory
    assert(slots.acknowledge(*handoff)&&slots.terminal_ready(ids[0]));
    assert(slots.release(ids[0]));
    auto reused=slots.admit(1,127,1,129);assert(reused);
    assert(reused->slot==ids[0].slot&&reused->generation!=ids[0].generation);
    assert(!slots.lookup(ids[0])&&!slots.acknowledge(*w)&&!slots.can_deliver(*w));
    assert(!slots.finish(ids[0],true)&&!slots.release(ids[0]));
    assert(slots.decode_ready(*reused)&&slots.commit_position(*reused,128));
    assert(!slots.commit_position(*reused,127)&&!slots.commit_position(*reused,130));
    assert(slots.lookup(ids[1])->phase==RequestPhase::prefill); // cancel isolation
    assert(slots.finish(*reused,false)&&slots.release(*reused));
    const auto max=std::numeric_limits<int64_t>::max();
    assert(!slots.admit(max,1,0,max)&&!slots.admit(-1,1,0,10));
    assert(!slots.admit(1,1,max,max)&&!slots.admit(0,0,0,0));
    assert(!yoco_cone_last_chunk(20,2,3,6720,6720-2667));
    assert(yoco_cone_last_chunk(20,2,3,6720,6720-2668));
    assert(!yoco_cone_last_chunk(20,2,3,6720,6720-2669));
}
'''
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    (p/'test.cpp').write_text(source)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',
                    '-I'+str(root/'src/dsv41'),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
    subprocess.run([str(p/'test')],check=True)
print('PASS host state ownership, view binding, four leases, cancellation/drain, stale/duplicate completion, generation reuse, overflow admission, cone 2667/2668/2669')
