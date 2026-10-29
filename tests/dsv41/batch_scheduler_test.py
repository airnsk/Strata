#!/usr/bin/env python3
"""Executes the actual native scheduler with a deterministic CPU model stub."""
from pathlib import Path
import os
import sys
spec_mode = "--spec" in sys.argv
import queue
import subprocess
import tempfile
import threading
import time

root=Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as d:
    exe=Path(d)/"mock"
    subprocess.run(["g++","-std=c++17","-O1","-g","-pthread","-Wall","-Wextra","-Werror",
                    "-I"+str(root/"src/dsv41"),str(root/"tests/dsv41/batch_scheduler_mock.cpp"),
                    "-o",str(exe)],check=True)
    proc=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           text=True,env={**os.environ,"DS4_BATCH_CAP":"128", "DS4_BATCH_SPEC":str(int(spec_mode))})
    q=queue.Queue()
    def pump():
        for line in proc.stdout:q.put(line.strip())
        q.put(None)
    threading.Thread(target=pump,daemon=True).start()
    def take_until(predicate,timeout=5):
        result=[];end=time.monotonic()+timeout
        while time.monotonic()<end:
            line=q.get(timeout=max(.01,end-time.monotonic()))
            assert line is not None,"unexpected EOF"
            result.append(line)
            if predicate(result):return result
        raise AssertionError(result)
    take_until(lambda xs:xs[-1].startswith("READY"))
    # One write makes all admissions visible together; no sequential GEN path.
    proc.stdin.write("".join(f"BGEN2 {b} 1 8 seed={b} spec={int(spec_mode)} 1,2,3\n" for b in range(4)))
    proc.stdin.flush()
    result=take_until(lambda xs:sum(x.startswith("BDONE2") for x in xs)==4)
    for b in range(4):
        tokens=[int(x.split()[3]) for x in result if x.startswith(f"BT2 {b} 1 ")]
        assert tokens==[10+b+i for i in range(8)],(b,tokens,result)
    proc.stdin.write("BSTOP2 0 1\nBGEN2 -1 2 4 1\nBGEN2 0 2 1 seed=7 1\n")
    proc.stdin.flush()
    reuse=take_until(lambda xs:any(x.startswith("BDONE2 0 2") for x in xs))
    assert [x for x in reuse if x.startswith("BT2")]==["BT2 0 2 17"],reuse
    proc.stdin.write("BGEN2 1 2 2 temperature=nan 1\n");proc.stdin.flush()
    bad=take_until(lambda xs:xs[-1].startswith("BERR2"))
    assert bad[-1]=="BERR2 1 2 invalid_or_unsupported_request",bad

    # Identical draw stream despite transport segmentation, inserted prompt and neighbors.
    proc.stdin.write("BGEN2 0 3 8 seed=13 counter=0 1,2\n");proc.stdin.flush()
    whole=take_until(lambda xs:any(x.startswith("BDONE2 0 3") for x in xs))
    full=[int(x.split()[3]) for x in whole if x.startswith("BT2 0 3")]
    segmented=[]
    for epoch,offset,count in [(4,0,3),(5,3,5)]:
        proc.stdin.write(f"BGEN2 0 {epoch} {count} seed=13 counter={offset} 1,2,99\n"
                        +f"BGEN2 2 {epoch} 2 seed=37 counter=9 7\n")
        proc.stdin.flush()
        part=take_until(lambda xs:sum(x.startswith("BDONE2") for x in xs)==2)
        segmented += [int(x.split()[3]) for x in part if x.startswith(f"BT2 0 {epoch} ")]
    assert full==segmented,(full,segmented)
    print("NATIVE_COUNTER_SEGMENTATION_PASS unchanged draws with inserted prompt and neighbor")

    # Actual scheduler: a long admission cannot monopolize ready decode.
    lengths=[1,96,80,47]
    proc.stdin.write("".join(f"BGEN2 {b} 10 {12 if b==0 else 8} seed={b} "
                            + ",".join(["1"]*lengths[b])+"\n" for b in range(4)))
    proc.stdin.flush()
    fair=take_until(lambda xs:sum(x.startswith("BDONE2") for x in xs)==4)
    for b in range(4):
        progress=[int(x.split()[3]) for x in fair if x.startswith(f"BPP2 {b} 10 ")]
        assert progress and progress[-1]==lengths[b],(b,progress)
        assert all(0<end-start<=16 for start,end in zip([0]+progress,progress)),progress
    a_done=next(i for i,x in enumerate(fair) if x.startswith("BDONE2 0 10 "))
    b_filled=next(i for i,x in enumerate(fair) if x.startswith("BPP2 1 10 96 "))
    assert a_done<b_filled,fair
    a_tokens=[i for i,x in enumerate(fair) if x.startswith("BT2 0 10 ")]
    busy_prefills=[i for i,x in enumerate(fair) if x.startswith(("BPP2 1 10 ","BPP2 2 10 ","BPP2 3 10 "))]
    assert any(a_tokens[0]<i<a_tokens[-1] for i in busy_prefills),fair
    print("ACTUAL_SCHEDULER_FAIR_PREFILL_PASS max16 rows; decode interleaves; long admission progresses")
    proc.stdin.write("QUIT\n");proc.stdin.flush()
    proc.wait(timeout=10)
    err=proc.stderr.read()
    assert proc.returncode==0,err
    assert ("MOCK_SPEC 4" if spec_mode else "MOCK_PACKED 4") in err,err
    print("ACTUAL_NATIVE_SCHEDULER_MOCK_PASS four streams; shared batch4; seeded output; stale cancel; reuse; malformed admission")
