"""Versioned DS4 AR slot mailboxes. No GPU or model state is owned here."""
import math
import os
import json
import time
import queue
import threading
from collections import deque

_trace_lock = threading.Lock()
def _segment_trace(event, rng, **fields):
    path = os.environ.get("STRATA_DS4_TEST_WIRE_LOG")
    if path:
        record = {"event":event, "time":time.monotonic(), "logical":rng["logical"],
                  "seed":rng["seed"], "counter":rng["next"], **fields}
        with _trace_lock, open(path+".segments.jsonl","a",encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")


class BatchWire:
    def __init__(self, slots, capacity, max_waiters=32):
        if not 1 <= slots <= 4 or capacity < 1:
            raise ValueError("invalid DS4 batch capability")
        self.cv = threading.Condition()
        self.epochs = [0] * slots
        self.active = [False] * slots
        self.terminal = [False] * slots
        self.queues = [queue.Queue(maxsize=capacity + capacity // 16 + 8) for _ in range(slots)]
        self.waiters = deque()
        self.max_waiters = max_waiters
        self.closed = False

    def enqueue(self):
        with self.cv:
            if self.closed:
                raise RuntimeError("batch process ended")
            if len(self.waiters) >= self.max_waiters:
                raise ValueError("batch admission queue is full")
            ticket = object()
            self.waiters.append(ticket)
            return ticket

    def abandon(self, ticket):
        with self.cv:
            if ticket in self.waiters:
                self.waiters.remove(ticket)
            self.cv.notify_all()

    def claim(self, ticket):
        with self.cv:
            if self.closed:
                raise RuntimeError("batch process ended")
            if not self.waiters or self.waiters[0] is not ticket:
                return None
            for slot, busy in enumerate(self.active):
                if not busy:
                    self.waiters.popleft()
                    self.epochs[slot] += 1
                    self.active[slot] = True
                    self.terminal[slot] = False
                    q = self.queues[slot]
                    while not q.empty():
                        q.get_nowait()
                    return slot, self.epochs[slot], q
            return None

    def route(self, line):
        fields = line.split()
        if not fields or fields[0] not in ("BT2", "BPP2", "BDONE2", "BERR2"):
            return False
        # Recognized malformed records are discarded, never sent to another queue.
        try:
            op = fields[0]
            slot, epoch = int(fields[1]), int(fields[2])
            if not 0 <= slot < len(self.epochs) or epoch <= 0:
                return True
            if op == "BT2":
                if len(fields) != 4 or int(fields[3]) < 0:
                    return True
            elif op == "BPP2":
                if len(fields) != 5 or not 0 <= int(fields[3]) <= int(fields[4]):
                    return True
            elif op == "BDONE2":
                if len(fields) != 7 or fields[3] not in ("stop", "length", "cancel", "nonfinite"):
                    return True
                if int(fields[4]) < 0 or any(not math.isfinite(float(x)) or float(x) < 0 for x in fields[5:]):
                    return True
            elif len(fields) != 4:
                return True
        except (ValueError, IndexError, OverflowError):
            return True
        with self.cv:
            if self.closed or not self.active[slot] or self.epochs[slot] != epoch or self.terminal[slot]:
                return True
            try:
                self.queues[slot].put_nowait(line)
            except queue.Full:
                # Fail closed. The owner must stop the engine; never drop a token silently.
                self._close_locked()
                return True
            if op in ("BDONE2", "BERR2"):
                self.terminal[slot] = True
                self.cv.notify_all()
        return True

    def release(self, slot, epoch):
        with self.cv:
            if not 0 <= slot < len(self.epochs) or epoch != self.epochs[slot] or not self.active[slot]:
                return False
            if not self.closed and not self.terminal[slot]:
                return False
            self.active[slot] = False
            self.cv.notify_all()
            return True

    def _close_locked(self):
        self.closed = True
        for q in self.queues:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        self.cv.notify_all()

    def close(self):
        with self.cv:
            self._close_locked()

def stream_request(engine, ids, max_new, sampling, cancel, embeddings, error_type):
    """All-slot AR contract: no legacy GEN migration, no shared control reader."""
    if embeddings:
        raise ValueError("DS4 batch v2 does not support images")
    sampling = sampling or {}
    capacity = int(engine.info["batch_context"])
    if (not ids or any(type(t) is not int or t < 0 for t in ids)
            or type(max_new) is not int or max_new < 1 or len(ids) + max_new > capacity):
        raise ValueError("request exceeds DS4 resident capacity or contains invalid tokens")
    # Use the existing supported sampler grammar, but never silently discard a
    # requested feature that this AR backend cannot implement.
    unsupported = ("logprobs", "top_logprobs", "logit_bias", "strata_tune",
                   "strata_prefix", "strata_img_keys", "experimental_speed_projection")
    if any(sampling.get(k) not in (None, False, {}, []) for k in unsupported):
        raise ValueError("unsupported DS4 batch sampling feature")
    if (sampling.get("repetition_penalty", 1) not in (None, 1)
            or sampling.get("frequency_penalty", 0) not in (None, 0)
            or sampling.get("presence_penalty", 0) not in (None, 0)):
        raise ValueError("DS4 batch does not support penalties")
    for key, low, high in (("temperature", 0, math.inf), ("top_p", 0, 1), ("min_p", 0, 1)):
        v = sampling.get(key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v) or v < low or v > high or (key == "top_p" and v == 0)):
            raise ValueError("invalid DS4 sampling parameter")
    seed = sampling.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError("invalid DS4 seed")
    keys = engine.sampling_keys(sampling)
    if seed == 0:
        keys += " seed=0"
    allowed = {"temperature", "top_p", "top_k", "min_p", "seed"}
    if any(part.split("=")[0] not in allowed for part in keys.split()):
        raise ValueError("unsupported DS4 batch command option")
    # One cancel/context object spans all transport segments of one HTTP request.
    # Only delivered model tokens commit draws; server-inserted prompt tokens and
    # queued tokens discarded when a segment closes do not advance this stream.
    rng = getattr(cancel, "_strata_ds4_rng", None)
    effective_seed = 0 if seed is None else seed
    if rng is None:
        rng = {"seed": effective_seed, "next": 0, "logical": str(time.monotonic_ns())}
        cancel._strata_ds4_rng = rng
    elif rng["seed"] != effective_seed:
        raise ValueError("DS4 logical request seed changed between segments")
    counter = rng["next"]
    if not 0 <= counter <= (2**64 - 1) - max_new:
        raise ValueError("DS4 logical request RNG counter overflow")
    if engine.info.get("batch_counter") in ("1", 1):
        keys += f" counter={counter}"
    elif counter:
        raise ValueError("DS4 engine cannot preserve segmented RNG counter")
    wire, proc = engine.batch_wire, engine.proc
    ticket = wire.enqueue()
    lease = None
    sent = False

    def send(text):
        with engine.wlock:
            if engine.proc is not proc or wire.closed:
                raise error_type("batch process changed")
            try:
                proc.stdin.write(text + "\n")
                proc.stdin.flush()
            except (OSError, AttributeError):
                raise error_type("batch input closed") from None

    try:
        while lease is None:
            if cancel.is_set():
                return
            try:
                lease = wire.claim(ticket)
            except RuntimeError as ex:
                raise error_type(str(ex)) from None
            if lease is None:
                with wire.cv:
                    wire.cv.wait(timeout=0.1)
                yield None
        slot, epoch, q = lease
        engine.last = {"generated": 0, "prompt_tokens": len(ids), "prompt_ms": 0,
                       "decode_ms": 0, "finish": "", "reused": 0}
        _segment_trace("begin",rng,slot=slot,epoch=epoch,prompt_n=len(ids),max_new=max_new)
        send(f"BGEN2 {slot} {epoch} {max_new}{keys} spec=0 " + ",".join(map(str, ids)))
        sent = True
        stopped = False
        while True:
            if cancel.is_set() and not stopped:
                send(f"BSTOP2 {slot} {epoch}")
                stopped = True
            if wire.closed:
                raise error_type("batch output closed")
            try:
                line = q.get(timeout=0.1)
            except queue.Empty:
                yield None
                continue
            if line is None:
                raise error_type("batch output closed")
            fields = line.split()
            if fields[0] == "BT2":
                engine.last = {**engine.last, "generated": engine.last["generated"] + 1}
                if not stopped:
                    rng["next"] += 1
                    _segment_trace("token",rng,slot=slot,epoch=epoch)
                    yield int(fields[3])
            elif fields[0] == "BPP2":
                cancel._strata_batch_progress = (int(fields[3]), int(fields[4]))
                # Separate diagnostic file: progress is not an RNG draw or segment.
                trace = os.environ.get("STRATA_DS4_TEST_WIRE_LOG")
                if trace:
                    record = {"time":time.monotonic(), "logical":rng["logical"],
                              "slot":slot, "epoch":epoch,
                              "done":int(fields[3]), "total":int(fields[4])}
                    with _trace_lock, open(trace+".progress.jsonl","a",encoding="utf-8") as f:
                        f.write(json.dumps(record)+"\n")
            elif fields[0] == "BERR2":
                raise ValueError("native batch admission: " + fields[3])
            elif fields[0] == "BDONE2":
                engine.last = {**engine.last, "finish": fields[3], "generated": int(fields[4]),
                               "prompt_ms": float(fields[5]), "decode_ms": float(fields[6])}
                if fields[3] == "nonfinite":
                    raise error_type("nonfinite batch logits")
                return
    finally:
        if sent:
            _segment_trace("end",rng,slot=slot,epoch=epoch,delivered=rng["next"]-counter)
        wire.abandon(ticket)
        if lease is not None:
            slot, epoch, q = lease
            if not wire.release(slot, epoch):
                if not sent:
                    # No command was accepted by this process; failed transport
                    # is global and must not make the slot reusable speculatively.
                    wire.close()
                    wire.release(slot, epoch)
                else:
                    try:
                        send(f"BSTOP2 {slot} {epoch}")
                    except error_type:
                        wire.close()
                    def retire():
                        while not wire.release(slot, epoch):
                            with wire.cv:
                                wire.cv.wait(timeout=0.25)
                    threading.Thread(target=retire, daemon=True).start()
