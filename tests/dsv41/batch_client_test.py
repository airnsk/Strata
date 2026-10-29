#!/usr/bin/env python3
"""Real StrataEngine client/pump against actual native scheduler with CPU stub."""
from pathlib import Path
import concurrent.futures
import subprocess
import sys
import tempfile
import threading
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from serve.server import StrataEngine
root=Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as td:
    d=Path(td);exe=d/"batch-mock"
    subprocess.run(["g++","-std=c++17","-O1","-pthread","-I"+str(root/"src/dsv41"),
                    str(root/"tests/dsv41/batch_scheduler_mock.cpp"),"-o",str(exe)],check=True)
    engine=StrataEngine(str(exe),[],cwd=str(root),log=str(d/"engine.log"),env={"DS4_BATCH_CAP":"128"})
    try:
        assert engine.info["batch_protocol"]=="ds4-v2"
        assert engine.batch==4 and engine.batch_wire is not None
        barrier=threading.Barrier(4)
        def run(seed):
            barrier.wait(timeout=5)
            out=[v for v in engine.generate_batched([1,2,3],24,{"seed":seed},threading.Event()) if v is not None]
            assert out==[10+(seed+i)%40 for i in range(24)],(seed,out)
            assert engine.last["generated"]==24 and engine.last["reused"]==0
            return len(out)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(run,range(4)))==[24]*4
        assert not any(engine.batch_wire.active)
        barrier5=threading.Barrier(5)
        def queued(seed):
            cancel=threading.Event();barrier5.wait(timeout=5);out=[]
            for token in engine.generate_batched([2,3],32,{"seed":seed},cancel):
                if token is not None:
                    out.append(token)
                    if seed==9:cancel.set()
            if seed!=9:assert out==[10+(seed+i)%40 for i in range(32)],(seed,out)
            return len(out)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            result=list(pool.map(queued,range(5,10)))
        assert result[:4]==[32]*4 and result[4]==1,result
        assert not any(engine.batch_wire.active)
        for _ in range(2):
            cancel=threading.Event()
            stream=engine.generate_batched([1,2],64,{"seed":9},cancel)
            token=next(x for x in stream if x is not None)
            assert token==19
            cancel.set()
            list(stream)
        assert not any(engine.batch_wire.active)
        print("REAL_CLIENT_NATIVE_MOCK_PASS 4 concurrent streams; request-local last; cancel/reuse; fifth waiter; no legacy migration")
    finally:
        engine.close()
