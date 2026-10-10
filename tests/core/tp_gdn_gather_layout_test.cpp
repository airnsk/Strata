#include "strata/kernels/tp_gdn_gather_layout.hpp"
#include "strata/core/tp_layer_layout.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace layout = strata::kernels::tp_gdn_gather_layout;
void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
int main(){
    try{
        constexpr uint32_t poison=0xc3d2e1f0;
        size_t words=0;
        for(bool y:{false,true})for(int t=1;t<=8;++t)for(int local_rank:{0,1}){
            const int half=layout::half_width(y),full=2*half;
            std::array<std::vector<uint32_t>,2> source;
            for(int rank=0;rank<2;++rank){source[rank].resize(t*half);
                for(int i=0;i<t*half;++i)source[rank][i]=(rank?0x7fe00000u:0x80000000u)|uint32_t(i+1);}
            std::vector<uint32_t> gathered(8*full+2,poison),hits(t*full);
            // Same source->destination addressing as the production kernel;
            // the independent oracle below uses canonical head ownership.
            for(int i=0;i<t*half;++i)for(int rank:{local_rank,1-local_rank}){
                const int dst=(i/half)*full+layout::column(y,rank,i%half);
                require(dst>=0&&dst<t*full,"out-of-range destination");
                gathered[1+dst]=source[rank][i];++hits[dst];
            }
            for(int i=0;i<t*full;++i){
                const int col=i%full;
                const int rank=y?(col/128%16)/8:col/1280;
                const int src_col=y?(col/2048)*1024+col%1024:col%1280;
                require(hits[i]==1,"missing or multiply owned word");
                require(gathered[1+i]==source[rank][i/full*half+src_col],"raw-word gather mismatch");++words;
            }
            require(gathered.front()==poison,"prefix guard changed");
            for(size_t i=1+t*full;i<gathered.size();++i)require(gathered[i]==poison,"inactive capacity/tail guard changed");
        }
        strata::core::ModelGeometry geometry;
        for(int rank=0;rank<2;++rank){const auto mapping=strata::core::tp2::gdn_rank_layout(geometry,rank);
            for(int col=0;col<3072;++col)require(layout::column(true,rank,col)==mapping.value_rows[col],"canonical GDN weight ownership differs");}
        std::cout<<"TP_GDN_GATHER_CPU_PASS checked_words="<<words<<" raw_bits=1 T=1..8 capture_tested=0 rccl_tested=0\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
