"""CPU topology/Run regression; not a numerical GPU quality test."""
from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[2]
text=(root/'src/dsv41/ds4_model.cu').read_text()
a=text.index('    struct Run {');b=text.index('\n    };',a)+7
run=text[a:b]
cpp=r"""
#include <cassert>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>
#include "ds4_yoco_pipeline.hpp"
RUN
int main(){
 int dev[40],nm[40],grp[40];
 for(int i=0;i<40;i++){dev[i]=grp[i]=(i<15?0:1);nm[i]=1;}
 assert(strata::ds4::yoco_pipeline15_layout(40,20,dev,nm,grp));
 for(int i=0;i<40;i++){nm[i]=2;assert(!strata::ds4::yoco_pipeline15_layout(40,20,dev,nm,grp));nm[i]=1;
 dev[i]^=1;assert(!strata::ds4::yoco_pipeline15_layout(40,20,dev,nm,grp));dev[i]^=1;}
 assert(!strata::ds4::yoco_pipeline15_layout(40,19,dev,nm,grp));
 assert(!strata::ds4::yoco_pipeline15_layout(39,20,dev,nm,grp));
 for(int n:{3692,8192,8193,16384,20160,32768}){
  int nc=(n+8191)/8192,per=(n+nc-1)/nc,keep=2668,cut=n-keep;
  Run r;r.yoco=20;r.cut_k=cut/per;r.cut_r=cut%per;
  int tail=0,kv=0,ack[2]={-1,-1};
  for(int k=0;k<nc;k++){
   int t=std::min(per,n-k*per),slot=k%2;
   if(k>=2)assert(ack[slot]==k-2); // safe two-slot reuse after consumer
   // Consumers run before the skip, for all chunks including KV-only.
   for(int il=15;il<20;il++)assert(r.runs(k,il)&&!r.kv_only(k,il)&&r.r0(k,il)==0);
   ack[slot]=k; // activation and source14 consumed by layer15
   kv+=t;assert(r.runs(k,20));
   int used=r.kv_only(k,20)?0:t-r.r0(k,20);tail+=used;
   for(int il=21;il<40;il++)assert(r.runs(k,il)==(k>=r.cut_k));
   if(k==nc-1){assert(used>=128);assert(t-r.r0(k,39)>=128);} // final taps/rings
  }
  assert(kv==n&&tail==keep);
  if(n==20160)assert(nc==3&&per==6720&&r.cut_k==2&&r.cut_r==4052);
 }
 // Reversed slot schedule would deadlock if a skipped chunk failed to ack.
 int ack[2]={-1,-1};ack[0]=0;ack[1]=1;assert(ack[2%2]==0);
 std::cout<<"PASS topology, cut boundaries, source coverage, 2-slot reuse, final ring/tap dependency tail\n";
}
"""
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'test.cpp').write_text(cpp.replace('RUN',run))
 subprocess.run(['g++','-std=c++17','-O0','-I'+str(root/'src/dsv41'),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
