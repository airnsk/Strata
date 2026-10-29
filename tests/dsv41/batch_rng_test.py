#!/usr/bin/env python3
"""Logical request RNG offsets; no GPU or quality claims."""
import io, sys, threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from serve.ds4_batch_protocol import BatchWire, stream_request
class Engine:
    info={"batch_context":"128","batch_counter":"1"}
    def __init__(self):
        self.batch_wire=BatchWire(4,128); self.wlock=threading.Lock()
        self.last={}; self.commands=[]
        self.proc=type("Proc",(),{})()
        self.proc.stdin=self
    def sampling_keys(self,s):return f" seed={s.get('seed',0)}"
    def write(self,line):
        self.commands.append(line); f=line.split(); b,g=f[1:3]
        if f[0]=="BGEN2":
            opts=dict(x.split("=",1) for x in f[4:] if "=" in x)
            start=int(opts.get("counter",0));seed=int(opts.get("seed",0));n=int(f[3])
            for i in range(n):self.batch_wire.route(f"BT2 {b} {g} {10+(seed+start+i)%40}")
            self.batch_wire.route(f"BDONE2 {b} {g} length {n} 0 0")
    def flush(self):pass
def run(e,ctx,n,ids=(1,2),seed=13):
    return [t for t in stream_request(e,list(ids),n,{"seed":seed},ctx,None,RuntimeError) if t is not None]
e=Engine(); whole=run(e,threading.Event(),8)
ctx=threading.Event(); part=run(e,ctx,3)
run(e,threading.Event(),4,seed=37)
part+=run(e,ctx,5,ids=(1,2,99,98,97))
assert part==whole and ctx._strata_ds4_rng["next"]==8
# Closing early commits only yielded draws, not already queued native tokens.
ctx=threading.Event();gen=stream_request(e,[1],8,{"seed":13},ctx,None,RuntimeError)
first=next(gen);gen.close()
assert ctx._strata_ds4_rng["next"]==1
assert [first]+run(e,ctx,7)==whole
try:run(e,ctx,1,seed=14)
except ValueError:pass
else:raise AssertionError("changed logical seed accepted")
ctx._strata_ds4_rng["next"]=2**64-1
try:run(e,ctx,1)
except ValueError:pass
else:raise AssertionError("counter overflow accepted")
print("HTTP_LOGICAL_RNG_PASS segmented/neighbor/injected tokens/early close/seed/overflow")
