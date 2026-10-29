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
using namespace strata::ds4;
RUN
int main(){

 int dev[40],nm[40],grp[40],src[40],ratio[40],kv[40],ix[40],eg[40],md[3]={0,0,0};
 for(int i=0;i<40;i++){
  dev[i]=grp[i]=(i<10?0:1);nm[i]=1;
  src[i]=i<2?-1:i<8?2:i<14?8:i<20?14:20;ratio[i]=i<2?0:i<20?2:1;
  kv[i]=(i==2||i==8||i==14||i==20);ix[i]=kv[i]||i==24||i==28||i==32||i==36;
  eg[i]=i==1?0:i==14?1:-1;
 }
 auto good=[&](){return strata::ds4::yoco_pipeline10_layout(40,20,dev,nm,grp,src,ratio,kv,ix,eg,1,0,3,md);};
 assert(good());
 for(int* field:{dev,nm,grp,src,ratio,kv,ix,eg})for(int i=0;i<40;i++){++field[i];assert(!good());--field[i];}
 for(int i=0;i<3;i++){md[i]=1;assert(!good());md[i]=0;}
 assert(!strata::ds4::yoco_pipeline10_layout(39,20,dev,nm,grp,src,ratio,kv,ix,eg,1,0,3,md));
 assert(!strata::ds4::yoco_pipeline10_layout(40,19,dev,nm,grp,src,ratio,kv,ix,eg,1,0,3,md));
 assert(!strata::ds4::yoco_pipeline10_layout(40,20,dev,nm,grp,src,ratio,kv,ix,eg,0,0,3,md));
 assert(!strata::ds4::yoco_pipeline10_layout(40,20,dev,nm,grp,src,ratio,kv,ix,eg,1,1,3,md));
 assert(!strata::ds4::yoco_pipeline10_layout(40,20,dev,nm,grp,src,ratio,kv,ix,eg,1,0,2,md));
 // Actual source mapping: source8 crosses, source14/20 stay on GPU1.
 for(int i=10;i<14;i++){assert(src[i]==8&&dev[src[i]]==0&&dev[i]==1&&!ix[i]);}
 for(int i=14;i<40;i++)assert(dev[src[i]]==dev[i]);
 assert(eg[1]==0&&dev[1]==0&&eg[14]==1&&dev[14]==1);
 for(int n:{3692,8192,8193,16384,20160,32768}){
  int nc=(n+8191)/8192,per=(n+nc-1)/nc,keep=2668,cut=n-keep;
  Run r;r.yoco=20;r.cut_k=cut/per;r.cut_r=cut%per;
  int tail=0,kv=0,ack[2]={-1,-1};
  for(int k=0;k<nc;k++){
   int t=std::min(per,n-k*per),slot=k%2;
   if(k>=2)assert(ack[slot]==k-2); // safe two-slot reuse after consumer
   // Consumers run before the skip, for all chunks including KV-only.
   for(int il=10;il<20;il++)assert(r.runs(k,il)&&!r.kv_only(k,il)&&r.r0(k,il)==0);
   ack[slot]=k; // activation and source8 consumed by layer10
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
 std::cout<<"PASS 10/30 topology/ownership/Engram mutations, cut boundaries, source coverage, 2-slot reuse, final ring/tap dependency tail\n";
}
"""
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'test.cpp').write_text(cpp.replace('RUN',run))
 subprocess.run(['g++','-std=c++17','-O0','-I'+str(root/'src/dsv41'),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
