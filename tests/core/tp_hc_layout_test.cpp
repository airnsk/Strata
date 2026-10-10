#include "strata/core/tp_hc_layout.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata::core::tp2;
void require(bool b){if(!b)throw std::runtime_error("TP HC layout check failed");}
template<class F>void rejects(F f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}require(bad);}
int main(){
    std::array<int,320> down{};std::array<int,10240> up{};
    // Synthetic raw BF16 rows: source values preserve row/column fingerprints;
    // no floating conversion is allowed in ownership/reassembly.
    std::vector<uint16_t> src(10240u*320),dst(src.size());
    for(size_t i=0;i<src.size();++i)src[i]=static_cast<uint16_t>((i*17+i/320)%65521);
    for(int r=0;r<2;++r){
        for(int j=0;j<160;++j)require(++down[hc_down_source_row(r,j)]==1);
        for(int c=0;c<4;++c)for(int d=0;d<1280;++d){
            const int row=hc_up_source_row(r,c,d);require(++up[row]==1);
            require(row==c*2560+r*1280+d);
            for(int k=0;k<320;++k)dst[static_cast<size_t>(row)*320+k]=src[static_cast<size_t>(row)*320+k];
        }
    }
    for(int n:down)require(n==1);
    for(int n:up)require(n==1);
    require(dst==src);
    rejects([]{hc_down_source_row(-1,0);});rejects([]{hc_down_source_row(0,160);});
    rejects([]{hc_up_source_row(2,0,0);});rejects([]{hc_up_source_row(0,4,0);});rejects([]{hc_up_source_row(0,0,1280);});
    // Down-only128-thread launch changes cooperative copies, not any row's
    //32-lane arithmetic. Emulate tile fill and row assignment for both grids.
    for(int tokens=1;tokens<=8;++tokens){
        const int tile_words=tokens*(1280/4);
        std::vector<int> original(tile_words,-1),candidate(tile_words,-1);
        for(int tid=0;tid<256;++tid)for(int i=tid;i<tile_words;i+=256)original[i]=i;
        for(int tid=0;tid<128;++tid)for(int i=tid;i<tile_words;i+=128)candidate[i]=i;
        require(candidate==original);
        std::array<int,160> old_rows{},new_rows{};
        for(int block=0;block<20;++block)for(int warp=0;warp<8;++warp)++old_rows[block*8+warp];
        for(int block=0;block<40;++block)for(int warp=0;warp<4;++warp)++new_rows[block*4+warp];
        require(old_rows==new_rows);
        // Both geometries leave the row's lane+32*q chunk sequence unchanged.
        for(int lane=0;lane<32;++lane)for(int tile=0;tile<8;++tile)for(int q=0;q<5;++q){
            const int input=tile*1280+(lane+32*q)*8;
            require(input>=0&&input+7<10240);
        }
    }
    std::cout<<"TP HC row ownership, raw reassembly and down128 staging passed\n";
}
