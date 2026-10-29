// Included by ds4_main.cpp after its request parser. Experimental AR-only,
// one execution owner. Versioned wire events never alias a reused slot.
int serve_batch4(strata::ds4::Model& m,const std::string& eos_list) {
    const int64_t capacity=std::getenv("DS4_BATCH_CAP")?std::atoll(std::getenv("DS4_BATCH_CAP")):32768;
    if(capacity<=0 || capacity>m.max_context()) {std::fprintf(stderr,"invalid batch capacity\n");return 1;}
    struct Slot {
        uint64_t handle=0,epoch=0,counter=0;
        Req req;
        int read=0,generated=0,next=-1;
        bool busy=false,decode=false,cancel=false;
        double born=0,first=0,prompt_ms=0;
    };
    std::array<Slot,4> slots;
    std::string err;
    for(auto& s:slots) if(!m.request_create(capacity,s.handle,err)) {
        std::fprintf(stderr,"batch allocation: %s\n",err.c_str());return 1;
    }
    for(const auto& l:m.info_lines())std::printf("%s\n",l.c_str());
    std::printf("INFO batch_slots=4 batch_groups=1 batch_protocol=ds4-v2 batch_migration=0 batch_spec=0 batch_context=%lld\n",(long long)capacity);
    std::printf("READY %lld stop\n",(long long)m.max_context());std::fflush(stdout);
    auto reader=std::make_shared<Lines>();
    Lines& in=*reader;in.start(reader);
    const auto eos=parse_ids(eos_list);
    int next_prefill=0;bool quit=false;
    auto terminal=[&](int b,const char* why)->bool {
        auto& s=slots[b];
        const double decode=s.first?now_ms()-s.first:0;
        // Completion and arena release precede BDONE2; this is a no-more-writes barrier.
        if(!m.request_release(s.handle,err))return false;
        s.handle=0;
        std::printf("BDONE2 %d %llu %s %d %.3f %.3f\n",b,(unsigned long long)s.epoch,
                    why,s.generated,s.prompt_ms,decode);std::fflush(stdout);
        s.busy=false;s.decode=false;s.cancel=false;
        return true;
    };
    std::vector<float> checked_logits; // correctness gate; explicit cost in prototype timings
    auto output=[&](int b)->bool {
        auto& s=slots[b];
        if(!m.request_select(s.handle,err))return false;
        if(!m.logits(checked_logits)) {err="batch logits transfer failed";return false;}
        if(checked_logits.empty() || std::any_of(checked_logits.begin(),checked_logits.end(),
                [](float value){return !std::isfinite(value);})) return terminal(b,"nonfinite");
        const int token=m.sample(s.req.temperature,s.req.top_p,s.req.top_k,s.req.min_p,s.req.seed,s.counter);
        if(token<0 || token>=m.vocab()) return terminal(b,"nonfinite");
        ++s.counter;++s.generated;s.next=token;
        if(!s.first)s.first=now_ms();
        std::printf("BT2 %d %llu %d\n",b,(unsigned long long)s.epoch,token);std::fflush(stdout);
        if(std::find(eos.begin(),eos.end(),token)!=eos.end())return terminal(b,"stop");
        if(s.generated>=s.req.max_new)return terminal(b,"length");
        s.decode=true;return true;
    };
    while(!quit) {
        std::deque<std::string> commands;
        {
            std::lock_guard<std::mutex> lock(in.mu);
            commands.swap(in.q);
            quit=in.eof;
        }
        for(const auto& line:commands) {
            if(line=="QUIT"){quit=true;break;}
            std::istringstream parser(line);
            std::string op,slot_text,epoch_text;parser>>op>>slot_text>>epoch_text;
            int b=-1;uint64_t epoch=0;
            try {
                size_t used=0;
                b=std::stoi(slot_text,&used);
                if(used!=slot_text.size()||b<0||b>=4)throw std::invalid_argument("slot");
                if(epoch_text.empty()||epoch_text[0]=='-')throw std::invalid_argument("epoch");
                epoch=std::stoull(epoch_text,&used);
                if(used!=epoch_text.size()||epoch==0)throw std::invalid_argument("epoch");
            } catch(...) {std::fprintf(stderr,"rejected malformed batch command\n");continue;}
            auto& s=slots[b];
            if(op=="BSTOP2") {
                if(s.busy&&epoch==s.epoch)s.cancel=true;
                continue;
            }
            if(op!="BGEN2") {
                std::printf("BERR2 %d %llu unsupported_command\n",b,(unsigned long long)epoch);std::fflush(stdout);continue;
            }
            if(s.busy && epoch==s.epoch) {
                std::fprintf(stderr,"ignored duplicate live BGEN2\n");continue;
            }
            if(s.busy||epoch<=s.epoch) {
                std::printf("BERR2 %d %llu occupied_or_stale\n",b,(unsigned long long)epoch);std::fflush(stdout);continue;
            }
            std::string rest;std::getline(parser,rest);Req request;
            if(!strata::ds4::parse_batch_request(rest,m.vocab(),capacity,request)) {
                std::printf("BERR2 %d %llu invalid_or_unsupported_request\n",b,(unsigned long long)epoch);std::fflush(stdout);continue;
            }
            if(!s.handle&&!m.request_create(capacity,s.handle,err)) {
                std::printf("BERR2 %d %llu capacity_unavailable\n",b,(unsigned long long)epoch);std::fflush(stdout);err.clear();continue;
            }
            s.epoch=epoch;s.req=std::move(request);s.req.spec=false;s.counter=0;
            s.read=s.generated=0;s.next=-1;s.busy=true;s.decode=s.cancel=false;
            s.born=now_ms();s.first=s.prompt_ms=0;
        }
        if(quit)break;
        for(int b=0;b<4;++b)if(slots[b].busy&&slots[b].cancel)
            if(!terminal(b,"cancel")){quit=true;break;}
        if(quit)break;
        // Ready decode always precedes admission work. At most one 16-token
        // prefill quantum per tick; round-robin admission prevents starvation.
        std::vector<uint64_t> handles;std::vector<int> tokens,owners;
        for(int b=0;b<4;++b)if(slots[b].busy&&slots[b].decode) {
            handles.push_back(slots[b].handle);tokens.push_back(slots[b].next);owners.push_back(b);
        }
        if(!handles.empty()) {
            if(!m.decode_batch(handles,tokens,err)){quit=true;break;}
            for(int b:owners)if(!output(b)){quit=true;break;}
        }
        if(quit)break;
        bool prefilled=false;
        for(int j=0;j<4;++j) {
            const int b=(next_prefill+j)%4;auto& s=slots[b];
            if(!s.busy||s.decode)continue;
            if(!m.request_select(s.handle,err)){quit=true;break;}
            const int n=std::min(16,(int)s.req.ids.size()-s.read);
            const double t=now_ms();
            const bool last=s.read+n==(int)s.req.ids.size();
            if(!m.feed(s.req.ids.data()+s.read,n,last?1:0,err)){quit=true;break;}
            s.prompt_ms+=now_ms()-t;s.read+=n;prefilled=true;next_prefill=(b+1)%4;
            std::printf("BPP2 %d %llu %d %zu\n",b,(unsigned long long)s.epoch,s.read,s.req.ids.size());std::fflush(stdout);
            if(last&&!output(b))quit=true;
            break;
        }
        if(!prefilled&&handles.empty()&&!quit) {
            std::unique_lock<std::mutex> lock(in.mu);
            in.cv.wait(lock,[&]{return !in.q.empty()||in.eof;});
        }
    }
    // On any execution failure, all affected requests fail closed. No partial
    // state is reused; process exits after the model joins its workers.
    if(!err.empty())std::fprintf(stderr,"batch execution failure: %s\n",err.c_str());
    return err.empty()?0:1;
}
