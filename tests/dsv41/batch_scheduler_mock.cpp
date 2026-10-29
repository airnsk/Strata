#include "ds4_batch_request.hpp"
#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <cassert>
struct Req {int64_t max_new=0;bool spec=true;std::vector<int> ids;
float temperature=0,top_p=1,min_p=0;int top_k=64;uint64_t seed=0,rng_counter=0;};
double now_ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
std::vector<int> parse_ids(const std::string&){return {};}
struct Lines {
 std::mutex mu;std::condition_variable cv;std::deque<std::string> q;bool eof=false;
 void start(std::shared_ptr<Lines> lifetime={}) {
  std::thread([this,lifetime]{
   std::string line;
   while(std::getline(std::cin,line)){
    std::lock_guard<std::mutex> lock(mu);q.push_back(line);cv.notify_all();if(line=="QUIT")break;
   }
   std::lock_guard<std::mutex> lock(mu);eof=true;cv.notify_all();
  }).detach();
 }
};
namespace strata::ds4 {
class Model {
 uint64_t next=0,active=0;std::unordered_map<uint64_t,int> positions;
public:
 int max_context()const{return 128;}int vocab()const{return 100;}
 std::vector<std::string> info_lines()const{return {};}
 bool request_create(int64_t,uint64_t& h,std::string&){h=++next;positions[h]=0;return true;}
 bool request_select(uint64_t h,std::string&){assert(positions.count(h));active=h;return true;}
 bool request_release(uint64_t h,std::string&){assert(positions.erase(h)==1);return true;}
 bool feed(const int*,int n,int,std::string&){positions.at(active)+=n;return true;}
 bool decode_batch(const std::vector<uint64_t>& h,const std::vector<int>& t,std::string&) {
  assert(h.size()==t.size());
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  for(auto id:h)++positions.at(id);
  std::fprintf(stderr,"MOCK_PACKED %zu\n",h.size());return true;
 }
 bool logits(std::vector<float>& values){values={1,2,3};return true;}
 int sample(float,float,int,float,uint64_t seed,uint64_t counter) {
  return 10+(int)((seed+counter)%40);
 }
};
}
#include "ds4_batch_serve.inc"
int main(){strata::ds4::Model m;return serve_batch4(m,"");}
