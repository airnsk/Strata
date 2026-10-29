"""CPU dependency oracle; no claim of GPU numerical parity."""
from pathlib import Path
import subprocess, tempfile, math
root=Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    p.joinpath('test.cpp').write_text(r'''#include <cassert>
#include <iostream>
#include "ds4_yoco_pipeline.hpp"
using namespace strata::ds4;
int main(){
 for(int n:{3692,8192,8193,16384,20160,32768}){
  int nc=(n+8191)/8192,per=(n+nc-1)/nc,last=n-(nc-1)*per;
  int cut=n-2668,ck=cut/per,cr=cut%per;
  bool active=yoco_cone_last_chunk(20,ck,nc,last,cr);
  assert(active==(ck==nc-1 && last>=2668));
  for(int l=20;l<40;l++){
   int r=yoco_row_begin(20,ck,cr,active,ck,l);
   assert(r==cr+(active?(l-20)*127:0));
   assert(yoco_row_begin(20,ck,cr,active,ck,19)==0);
   if(active) assert(last-r==2668-(l-20)*127);
  }
 }
 assert(!yoco_cone_last_chunk(20,0,2,2500,0));
 assert(!yoco_cone_last_chunk(19,0,1,2668,0));
 assert(!yoco_cone_last_chunk(20,0,1,2668,-1));
 for(int l=20;l<40;l++)std::cout<<yoco_row_begin(20,2,4052,true,2,l)<<" ";
}
''')
    subprocess.run(['g++','-std=c++17','-I'+str(root/'src/dsv41'),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
    starts=list(map(int,subprocess.check_output([str(p/'test')],text=True).split()))
assert [6720-x for x in starts]==[2668-127*i for i in range(20)]
assert sum(6720-x for x in starts)==29230
assert sum(math.ceil((6720-x)/256) for x in starts)==120
assert sum(6720-starts[i] for i in [0,4,8,12,16])==8260
assert sum(math.ceil((6720-starts[i])/32) for i in [0,4,8,12,16])==260
# Modular nonlinear window plus row-specific far-history selections. Exact
# integers remove floating-point noise from this dependency-only test.
P=1000000007
for pos in [0,731]:
    n=3000; end=pos+n
    source={pos+i:(i*i+19*i+3)%P for i in range(n)}
    source_before=source.copy()
    full={t:(v,(v*7+5)%P) for t,v in source.items()}
    cone=full.copy(); full_sel={}; cone_sel={};full_cand={};cone_cand={}
    for layer in range(20,40):
        start=end-(2668-127*(layer-20))
        if layer in (37,38,39):
            assert [full[t] for t in range(end-128,end)]==[cone[t] for t in range(end-128,end)]
        # Current ring and the snapshot one position earlier use layer inputs.
        assert [full[t] for t in range(end-129,end)]==[cone[t] for t in range(end-129,end)]
        def step(inp, begin, sels, cands):
            out={}
            for t in range(begin,end):
                h,pre=inp[t]
                if layer==20:cands[t]=pos+(h+pre+t)%n
                if layer%4==0:sels[t]=pos+(h*3+pre+cands[t])%n
                # Before the view, poison represents absent local KV. Such
                # outputs are intentionally discarded by the next layer.
                z=sum((inp[q][0]*inp[q][0]+inp[q][1])%P if q>=begin else 987654321
                      for q in range(max(pos,t-127),t+1))%P
                v=(h*h+z+source[sels[t]]+layer)%P
                out[t]=(v,(pre+v*v)%P)
            return out
        full=step(full,pos,full_sel,full_cand)
        cone=step(cone,start,cone_sel,cone_cand)
        assert all(full[t]==cone[t] for t in range(start+127,end))
    assert all(full[t]==cone[t] for t in range(end-128,end))
    assert source==source_before
# All mandatory consumer chunks still acknowledge both physical slots.
ack=[-1,-1]
for k in range(7):
    if k>=2:assert ack[k%2]==k-2
    ack[k%2]=k
print('PASS actual range helpers, counts, nonlinear poisoned-window oracle, absolute selections, rings/snapshot/taps, source history, slot schedule')
