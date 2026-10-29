#!/usr/bin/env python3
"""CPU protocol/strict-parser checks; not evidence of packed GPU execution."""
from pathlib import Path
import io
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.ds4_batch_protocol import BatchWire, stream_request


class WireTests(unittest.TestCase):
    def lease(self, wire):
        return wire.claim(wire.enqueue())

    def test_generation_terminal_and_bounds(self):
        w=BatchWire(4,128)
        leases=[self.lease(w) for _ in range(4)]
        t=w.enqueue()
        self.assertIsNone(w.claim(t))
        b,e,q=leases[0]
        self.assertFalse(w.release(b,e))
        for line in ("BT2 -1 1 55", "BDONE2 4 1 stop 1 0 0", "BT2 x 1 7",
                     "BT2 0 1 -1", "BDONE2 0 1 length 1 nan 0"):
            self.assertTrue(w.route(line))
        self.assertTrue(all(x[2].empty() for x in leases))
        w.route("BT2 0 1 7")
        self.assertEqual(q.get_nowait(),"BT2 0 1 7")
        w.route("BDONE2 0 1 cancel 1 2 3")
        w.route("BT2 0 1 8")
        self.assertEqual(q.qsize(),1)
        self.assertTrue(w.release(b,e))
        b2,e2,q2=w.claim(t)
        self.assertEqual((b2,e2),(0,2))
        w.route("BT2 0 1 99")
        w.route("BDONE2 0 1 stop 8 1 2")
        self.assertTrue(q2.empty())
        self.assertFalse(w.release(0,1))
        w.route("BT2 0 2 10")
        self.assertEqual(q2.get_nowait(),"BT2 0 2 10")

    def test_fifo_abandon_eof_and_overflow(self):
        w=BatchWire(1,1,max_waiters=2)
        b,e,q=self.lease(w)
        a=w.enqueue();c=w.enqueue()
        with self.assertRaises(ValueError):w.enqueue()
        w.abandon(a)
        self.assertIsNone(w.claim(c))
        w.close()
        self.assertIsNone(q.get_nowait())
        self.assertTrue(w.release(b,e))
        with self.assertRaises(RuntimeError):w.claim(c)
        w=BatchWire(1,1);b,e,q=self.lease(w)
        for _ in range(10):w.route("BT2 0 1 1")
        self.assertTrue(w.closed)

    def test_stream_contract_and_cancel_reuse(self):
        class Dead(RuntimeError):pass
        class Engine:pass
        e=Engine();e.info={"batch_context":"128"};e.batch_wire=BatchWire(4,128)
        e.wlock=threading.Lock();e.last={};e.progress=None
        e.sampling_keys=lambda s: ""
        class Input:
            def __init__(self):self.commands=[]
            def write(self,line):
                self.commands.append(line)
                f=line.split();b,g=f[1:3]
                if f[0]=="BGEN2":
                    e.batch_wire.route(f"BPP2 {b} {g} 2 2")
                    e.batch_wire.route(f"BT2 {b} {g} 12")
                elif f[0]=="BSTOP2":
                    e.batch_wire.route(f"BDONE2 {b} {g} cancel 1 1 2")
            def flush(self):pass
        class Proc:pass
        e.proc=Proc();e.proc.stdin=Input()
        gen=stream_request(e,[1,2],8,{},threading.Event(),None,Dead)
        self.assertEqual(next(gen),12)
        gen.close()
        # Terminal was routed synchronously, background retire must finish.
        with e.batch_wire.cv:
            e.batch_wire.cv.wait_for(lambda:not any(e.batch_wire.active),timeout=2)
        self.assertFalse(any(e.batch_wire.active))
        self.assertTrue(e.proc.stdin.commands[0].startswith("BGEN2 0 1 "))
        gen2=stream_request(e,[1,2],8,{"seed":0},threading.Event(),None,Dead)
        self.assertEqual(next(gen2),12)
        self.assertIn("seed=0",e.proc.stdin.commands[-1])
        e.batch_wire.route("BDONE2 0 1 stop 99 0 0")
        e.batch_wire.route("BDONE2 0 2 length 1 1 2")
        self.assertEqual(list(gen2),[])
        self.assertEqual(e.last["generated"],1)
        self.assertFalse(any(e.batch_wire.active))
        for opts in ({"temperature":float("nan")},{"logit_bias":{"1":1}},{"seed":-1},
                     {"top_p":0},{"frequency_penalty":1}):
            with self.assertRaises(ValueError):
                list(stream_request(e,[1,2],8,opts,threading.Event(),None,Dead))
        self.assertEqual(len(e.proc.stdin.commands),3)


def native_parser():
    root=Path(__file__).resolve().parents[2]
    src=r"""
#include "ds4_batch_request.hpp"
#include <cassert>
#include <vector>
struct Req {int64_t max_new=0;bool spec=true;std::vector<int> ids;
float temperature=0,top_p=1,min_p=0;int top_k=64;uint64_t seed=0;};
int main(){
Req r;using strata::ds4::parse_batch_request;
assert(parse_batch_request("2 seed=0 temperature=0 top_p=1 spec=0 1,2",20,4,r));
assert(r.ids.size()==2&&r.max_new==2&&!r.spec);
for(auto s:{"0 1","2 1,","2 ,1","2 1x","2 20","2 -1","2 1 2","2 seed=-1 1",
"2 seed=18446744073709551616 1","2 temperature=nan 1","2 temperature=inf 1",
"2 top_p=0 1","2 top_k=65 1","2 unknown=1 1","2 spec=1 1","2 seed=1 seed=2 1",
"3 1,2","2 1,,2","2 temperature=1garbage 1"}) {
Req v;assert(!parse_batch_request(s,20,4,v));
}
}
"""
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);(p/"test.cpp").write_text(src)
        subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror",
                        "-fsanitize=address,undefined","-I"+str(root/"src/dsv41"),
                        str(p/"test.cpp"),"-o",str(p/"test")],check=True)
        subprocess.run([str(p/"test")],check=True)


if __name__=="__main__":
    native_parser()
    unittest.main()
