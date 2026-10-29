#pragma once
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
namespace strata::ds4 {
inline bool batch_u64(const std::string& s,uint64_t& v) {
    if(s.empty())return false;
    for(char c:s)if(c<'0'||c>'9')return false;
    errno=0;char* end=nullptr;
    v=std::strtoull(s.c_str(),&end,10);
    return errno!=ERANGE&&end==s.c_str()+s.size();
}
inline bool batch_float(const std::string& s,float& v) {
    if(s.empty())return false;
    errno=0;char* end=nullptr;v=std::strtof(s.c_str(),&end);
    return errno!=ERANGE&&end==s.c_str()+s.size()&&std::isfinite(v);
}
// Strict, bounded text-only AR grammar. Validate before allocating/mutating state.
template<class Request>
bool parse_batch_request(const std::string& text,int vocab,int64_t capacity,Request& r) {
    std::istringstream in(text);std::string field;uint64_t count=0;
    if(!(in>>field)||!batch_u64(field,count)||count==0||count>(uint64_t)capacity)return false;
    r.max_new=(int64_t)count;r.spec=false;r.ids.clear();r.rng_counter=0;
    std::unordered_set<std::string> keys;
    bool ids=false;
    while(in>>field) {
        if(ids)return false;
        auto eq=field.find('=');
        if(eq!=std::string::npos) {
            const auto k=field.substr(0,eq),v=field.substr(eq+1);
            if(!keys.insert(k).second)return false;
            uint64_t n=0;float f=0;
            if(k=="seed"){if(!batch_u64(v,n))return false;r.seed=n;}
            else if(k=="counter"){if(!batch_u64(v,n))return false;r.rng_counter=n;}
            else if(k=="spec"){if(v!="0")return false;}
            else if(k=="top_k"){if(!batch_u64(v,n)||n>64)return false;r.top_k=(int)n;}
            else if(k=="temperature"){if(!batch_float(v,f)||f<0)return false;r.temperature=f;}
            else if(k=="top_p"){if(!batch_float(v,f)||f<=0||f>1)return false;r.top_p=f;}
            else if(k=="min_p"){if(!batch_float(v,f)||f<0||f>1)return false;r.min_p=f;}
            else return false;
        } else {
            ids=true;size_t begin=0;
            while(begin<field.size()) {
                size_t end=field.find(',',begin);
                if(end==std::string::npos)end=field.size();
                uint64_t token=0;
                if(!batch_u64(field.substr(begin,end-begin),token)||token>=(uint64_t)vocab)return false;
                r.ids.push_back((int)token);
                if((uint64_t)r.ids.size()+count>(uint64_t)capacity)return false;
                if(end==field.size())break;
                begin=end+1;if(begin==field.size())return false;
            }
        }
    }
    return ids&&!r.ids.empty() && r.rng_counter <= std::numeric_limits<uint64_t>::max()-count;
}
}
