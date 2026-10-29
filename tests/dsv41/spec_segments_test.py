from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[2]
source=r'''#include "ds4_spec_segments.hpp"
#include <cassert>
#include <climits>
using namespace strata::ds4;
int main(){
 std::vector<size_t> ends;
 for(auto rows: {std::vector<int>{15}, {16}, {8,8}, {6,6,6,6}, {8,9}, {1,1,1,1}}) {
  assert(spec_segment_ends(rows,16,ends));
  size_t first=0;
  for(auto end:ends){int sum=0;assert(end>first);for(size_t i=first;i<end;++i)sum+=rows[i];assert(sum<=16);first=end;}
  assert(first==rows.size());
 }
 assert(spec_segment_ends({6,6,6,6},16,ends));assert((ends==std::vector<size_t>{2,4}));
 assert(spec_segment_ends({8,9},16,ends));assert((ends==std::vector<size_t>{1,2}));
 for(auto bad:{std::vector<int>{}, {17}, {0}, {-1}, {INT_MAX}, {1,1,1,1,1}}){
  assert(!spec_segment_ends(bad,16,ends));assert(ends.empty());
 }
 assert(!spec_segment_ends({1},0,ends));
}'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'test.cpp').write_text(source)
 subprocess.run(['c++','-std=c++20','-fsanitize=address,undefined','-I'+str(root/'src/dsv41'),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
print('SPEC_SEGMENTS_PASS boundaries15/16/17/24; whole segments; invalid/overflow')
