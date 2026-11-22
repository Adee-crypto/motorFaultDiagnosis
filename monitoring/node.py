"""
node.py
========
A simulated monitoring node: a program that pretends to be the small
computer attached to one motor.

    python monitoring/node.py --node-id node-1 --motor-id pump-A --data Loaded/RU --load 1
    python monitoring/node.py --node-id node-2 --motor-id fan-B --data Unloaded/HH --load 0 --realtime --speed 5
    python monitoring/node.py --node-id node-1 --motor-id pump-A --data Loaded/RU --load 1 --broker localhost
    python monitoring/node.py --node-id node-3 --motor-id pump-C --data Loaded/FB --load 1 --broker localhost --cycles 0 --realtime --speed 5 --fail-after 40 --fail-mode crash

What it does
  1. "Sensors": replays the motor's CSV recordings one second at a time,
     the way live sensors would deliver them (SensorReplay). With
     --realtime it also waits, so a 10 s run takes 10 s (or 10/speed s).
  2. After each run: works out the features and the Stage A check
     (is this run above the healthy level?) and saves a "run_result".
  3. After a full cycle of runs (normally all 8 speed profiles): gives the
     three verdicts (triggers, ML, combined) and saves a "verdict".
  4. Everything is saved in the node's local store (local_store.py) first.
  5. With --broker, each message is then sent through the MQTT broker
     (messaging.py) and marked "sent" once the broker confirms it. If the
     broker can't be reached, the node keeps working and the messages stay
     unsent in the local store.
  6. With --broker, it also sends a heartbeat ("I'm alive" and what it is
     doing) every few seconds, and the broker holds a "last will" that
     announces the node offline if its connection dies without a goodbye
     (messaging.py). The server uses both to spot failed nodes (step 4).
  7. Store-and-forward (step 5): results that could not be sent (broker
     down, network cut, crash before sending) are sent automatically, oldest
     first, as soon as the connection is back. When the server asks for
     message numbers it is missing, the node sends them again from its
     store, or says it no longer has them. Only one program can use a
     node's store at a time (it is locked while the node runs).

The diagnosis itself is the unchanged code in the folder above
(predict_motor.py, ml/predict_ml.py, fault_feature_extraction.py), so a
node gives exactly the same verdicts as predict_all.py.

Options
  --data        folder with the motor's run CSVs (X_X_<1-8>_<load>.csv)
  --load        0 = unloaded, 1 = loaded
  --store       the node's database file (default monitoring/stores/<node-id>.db)
  --cycles      how many full cycles to run (default 1; 0 = forever)
  --realtime    replay at the recording's real pace instead of as fast as possible
  --speed       with --realtime: playback speed-up (default 1)
  --broker      MQTT broker as host or host:port (e.g. localhost); without it
                the node only saves locally, as in step 1
  --heartbeat   seconds between heartbeats (default 5)
  --flush-wait  at the end, seconds to keep trying to send unsent results (default 10)
  --fail-after, --fail-mode   for testing failure detection: after this many
                seconds, either "crash" (the program dies at once, no goodbye)
                or "hang" (it freezes: stays connected but stops working and
                stops sending heartbeats)
"""
import argparse
import datetime as dt
import glob
import os
import re
import signal
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "ml"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import fault_feature_extraction as F          # noqa: E402
import predict_motor as PM                    # noqa: E402  hand-found triggers
import predict_ml as PML                      # noqa: E402  ML-found triggers
from process_fault import FALLBACK_FS, CHANNEL_MAP, SPEED_LABELS  # noqa: E402
from local_store import LocalStore, StoreInUse  # noqa: E402
from messaging import Publisher               # noqa: E402

warnings.filterwarnings("ignore")


def now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def jsonable(x):
    """numpy numbers and bools -> plain Python, so messages can be saved as JSON"""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 4)
    return x


class SensorReplay:
    """Delivers one recording in 1 s blocks, like live sensors would."""

    def __init__(self, path, block_seconds=1.0, realtime=False, speed=1.0):
        self.path, self.realtime, self.speed = path, realtime, speed
        self.block_rows = int(round(FALLBACK_FS * block_seconds))
        self.block_seconds = block_seconds

    def blocks(self):
        t0 = time.monotonic()
        for i, chunk in enumerate(pd.read_csv(self.path, encoding="utf-8-sig", chunksize=self.block_rows)):
            chunk = chunk.loc[:, ~chunk.columns.str.contains("^Unnamed")].dropna(how="all")
            if self.realtime:                      # wait until this block would really have arrived
                due = t0 + (i + 1) * self.block_seconds / self.speed
                time.sleep(max(0.0, due - time.monotonic()))
            yield chunk


class Node:
    def __init__(self, node_id, motor_id, data_dir, load, store_path, realtime=False, speed=1.0, broker=None,
                 heartbeat=5.0, fail_after=None, fail_mode=None, flush_wait=10.0):
        self.node_id, self.motor_id, self.load = node_id, motor_id, load
        self.realtime, self.speed = realtime, speed
        self.files = self._find_runs(data_dir, load)
        try:
            self.store = LocalStore(store_path, node_id, motor_id, lock=True)
        except StoreInUse:
            sys.exit(f"{node_id} is already running on this computer (its store {store_path} is in use). "
                     f"Stop the other one first, or use a different --node-id.")
        self.flush_wait = flush_wait
        self.seq = self.store.next_seq()
        self.healthy = PM.read_calibration(
            os.path.join(ROOT, f"trigger_calibration_final_load{load}.json"))["healthy_reference"]
        self.stop = False
        # what the node is doing, reported in its heartbeats
        self.state, self.cycle, self.runs_done = "starting", 0, 0
        self.unsent_count = len(self.store.unsent())   # kept in memory: the heartbeat thread can't use the database
        self.started = time.monotonic()
        self.heartbeat_s, self.fail_after, self.fail_mode = heartbeat, fail_after, fail_mode
        self.frozen = False
        self._hb_stop = threading.Event()
        self.publisher = Publisher(broker, client_id=node_id, log=self.log, motor_id=motor_id) if broker else None

    @staticmethod
    def _find_runs(data_dir, load):
        runs = {}
        for f in glob.glob(os.path.join(data_dir, "*.csv")):
            m = re.search(r"_(\d)_(\d)\.csv$", os.path.basename(f))
            if m and int(m.group(2)) == load:
                runs[int(m.group(1))] = f
        if not runs:
            sys.exit(f"no X_X_<1-8>_{load}.csv files in {data_dir}")
        return [runs[c] for c in sorted(runs)]    # the motor steps through its speed profiles in order

    def log(self, text):
        print(f"[{dt.datetime.now():%H:%M:%S}] {self.node_id} ({self.motor_id}): {text}", flush=True)

    def emit(self, type_, payload):
        """build a message, save it locally, then send it if a broker is set"""
        msg = {"msg_id": f"{self.node_id}-{self.store.epoch}-{self.seq}", "seq": self.seq,
               "epoch": self.store.epoch, "type": type_,
               "node_id": self.node_id, "motor_id": self.motor_id, "load": self.load,
               "created_at": now_utc(), "payload": jsonable(payload)}
        self.store.add(msg)                       # saved first: nothing is lost if sending fails
        self.seq += 1
        self.unsent_count += 1
        if self.publisher:
            self._service(fresh=True)             # sends the backlog first, then this one
            if self.unsent_count:
                self.log(f"message #{msg['seq']} not sent; kept locally ({self.unsent_count} waiting)")
        return msg

    # ------------------------------------------------ store-and-forward (step 5)
    def _service(self, fresh=False):
        """send unsent results (oldest first) and answer the server's resend requests;
        main thread only, because it uses the store"""
        if not self.publisher or not self.publisher.connected.is_set():
            return
        self.publisher.reconnected.clear()
        if self.unsent_count:
            sent = []
            for m in self.store.unsent():
                if not self.publisher.send(m):
                    break
                self.store.mark_sent(m["seq"])
                sent.append(m["seq"])
            self.unsent_count = len(self.store.unsent())
            # the message just created is the newest one; everything sent before it was waiting
            older = [q for q in sent if q != self.seq - 1] if fresh else sent
            if older:
                self.log(f"sent backlog: {len(older)} result(s) kept from earlier ({_ranges(older)})")
        while not self.publisher.requests.empty():
            self._answer(self.publisher.requests.get())

    def _answer(self, req):
        seqs = sorted(set(int(q) for q in req.get("seqs", [])))
        if req.get("epoch", "") != self.store.epoch:
            self.publisher.reply({"epoch": req.get("epoch", ""), "unavailable": seqs,
                                  "reason": "the node no longer has that store (it was replaced or wiped)"})
            self.log(f"server asked for {len(seqs)} message(s) from an old store (epoch {req.get('epoch')}): "
                     f"told it they are gone")
            return
        resent, gone = [], []
        for q in seqs:
            m = self.store.get(q)
            if m is None:
                gone.append(q)
            elif self.publisher.send(m):
                self.store.mark_sent(q)
                resent.append(q)
        if gone:
            self.publisher.reply({"epoch": self.store.epoch, "unavailable": gone,
                                  "reason": "not in the node's store"})
        self.unsent_count = len(self.store.unsent())
        self.log(f"server asked for {len(seqs)} missing message(s): resent {_ranges(resent) or 'none'}"
                 + (f", no longer have {_ranges(gone)}" if gone else ""))

    # ------------------------------------------------ failure detection (step 4)
    def _heartbeat_loop(self):
        while not self._hb_stop.wait(self.heartbeat_s):
            if self.frozen:                       # a frozen program sends nothing
                continue
            self.publisher.heartbeat({"state": self.state, "cycle": self.cycle, "runs_done": self.runs_done,
                                      "runs_per_cycle": len(self.files), "unsent": self.unsent_count,
                                      "uptime_s": round(time.monotonic() - self.started), "interval_s": self.heartbeat_s})

    def _inject_failure(self):
        if self.fail_mode == "crash":
            self.log(f"TEST: crashing now (after {self.fail_after:g} s): no goodbye, no more heartbeats")
            os._exit(1)                           # dies instantly, like a power cut or a crashed program
        self.log(f"TEST: freezing now (after {self.fail_after:g} s): still connected, but no work and no heartbeats")
        self.frozen = True

    def _wait_if_frozen(self):
        while self.frozen and not self.stop:
            time.sleep(0.5)

    def record_run(self, path):
        """receive one run from the sensors, block by block; None if stopped"""
        blocks = []
        for b in SensorReplay(path, realtime=self.realtime, speed=self.speed).blocks():
            self._wait_if_frozen()
            if self.stop:
                return None
            blocks.append(b)
        df = pd.concat(blocks, ignore_index=True)
        df.insert(0, "Time", np.arange(len(df)) / FALLBACK_FS)
        return df

    def run_cycle(self, cycle):
        vectors = {}
        self.cycle, self.runs_done = cycle, 0
        for path in self.files:
            cid = PM.speed_of(path, None)
            self.state = f"recording {os.path.basename(path)}"
            df = self.record_run(path)
            if df is None:
                return False
            self.state = f"analysing {os.path.basename(path)}"
            vec = F.build_feature_vector(df, FALLBACK_FS, channel_map=CHANNEL_MAP, condition_id=cid)
            vectors[path] = vec
            faulty, ev = F.is_faulty(vec, cid, self.healthy.get(cid))
            a1 = ev.get("accel1__stat_variance_ratio_vs_healthy")
            mic = ev.get("mic__stat_variance_ratio_vs_healthy")
            self.emit("run_result", {"cycle": cycle, "run": os.path.basename(path), "condition_id": cid,
                                     "speed": SPEED_LABELS[cid], "faulty": faulty,
                                     "accel1_x_healthy": a1, "mic_x_healthy": mic})
            self.log(f"run {os.path.basename(path)} [{SPEED_LABELS[cid]}] "
                     f"{'FAULTY' if faulty else 'healthy level'} (accel1 {a1:.1f}x, mic {mic:.1f}x)")
            self.runs_done += 1
            self._wait_if_frozen()
            self._service()
            if self.stop:
                return False

        files = list(vectors)
        self.state = "diagnosing"
        hand = PM.diagnose(files, self.load, verbose=False, vectors=vectors)
        ml = PML.diagnose_ml(files, self.load, verbose=False, vectors=vectors)
        pts = {k: hand["points"].get(k, 0.0) + ml["points"].get(k, 0.0)
               for k in set(hand["points"]) | set(ml["points"])}
        verdict, conf = PML.verdict_from_points(pts, hand["faulty_runs"] > 0)
        out = {"cycle": cycle, "runs": len(files), "faulty_runs": hand["faulty_runs"],
               "triggers": {"verdict": hand["verdict"], "confidence": hand["confidence"], "points": hand["points"]},
               "ml": {"verdict": ml["verdict"], "confidence": ml["confidence"], "points": ml["points"]},
               "combined": {"verdict": verdict, "confidence": conf,
                            "points": dict(sorted(pts.items(), key=lambda kv: -kv[1]))},
               "agree": hand["verdict"] == ml["verdict"]}
        self.emit("verdict", out)
        self.state = "idle"
        self.log(f"cycle {cycle} done: triggers {hand['verdict']} ({hand['confidence']}), "
                 f"ML {ml['verdict']} ({ml['confidence']}), combined {verdict} ({conf})")
        return True

    def run(self, cycles=1):
        self.log(f"start: {len(self.files)} runs per cycle, load {self.load}, "
                 f"{'real time x' + str(self.speed) if self.realtime else 'fast replay'}, "
                 f"next message #{self.seq} (store epoch {self.store.epoch})")
        if self.publisher:
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
            self.log(f"heartbeat every {self.heartbeat_s:g} s")
            if self.unsent_count:
                self.log(f"{self.unsent_count} result(s) from earlier not sent yet; sending them first")
                self._service()
        if self.fail_mode:
            t = threading.Timer(self.fail_after, self._inject_failure)
            t.daemon = True
            t.start()
        c = 1
        while not self.stop and (cycles == 0 or c <= cycles):
            if not self.run_cycle(c):
                break
            c += 1
        if self.publisher and not self.frozen:
            deadline = time.monotonic() + self.flush_wait     # last chance to deliver the backlog
            while self.unsent_count and time.monotonic() < deadline:
                self._service()
                time.sleep(0.5)
            self._service()
        unsent = len(self.store.unsent())
        self.log(("stopped" if self.stop else "finished") + (f"; {unsent} message(s) not sent, kept for next time"
                                                              if unsent else ""))
        self._hb_stop.set()
        if self.publisher:
            self.publisher.close("stopped by user" if self.stop else "finished")
        self.store.close()


def _ranges(nums):
    """[1,2,3,7] -> '#1-#3, #7'"""
    out, nums = [], sorted(nums)
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"#{nums[i]}" if i == j else f"#{nums[i]}-#{nums[j]}")
        i = j + 1
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node-id", required=True)
    ap.add_argument("--motor-id", required=True)
    ap.add_argument("--data", required=True, help="folder with the motor's run CSVs")
    ap.add_argument("--load", type=int, choices=[0, 1], required=True)
    ap.add_argument("--store", help="database file (default monitoring/stores/<node-id>.db)")
    ap.add_argument("--cycles", type=int, default=1, help="full cycles to run; 0 = forever")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--broker", help="MQTT broker, host or host:port (e.g. localhost)")
    ap.add_argument("--heartbeat", type=float, default=5.0, help="seconds between heartbeats (default 5)")
    ap.add_argument("--flush-wait", type=float, default=10.0,
                    help="at the end, seconds to keep trying to send unsent results (default 10)")
    ap.add_argument("--fail-after", type=float, help="TEST: seconds until a simulated failure")
    ap.add_argument("--fail-mode", choices=["crash", "hang"], help="TEST: crash = die at once; hang = freeze")
    a = ap.parse_args()
    store = a.store or os.path.join(HERE, "stores", f"{a.node_id}.db")
    if (a.fail_after is None) != (a.fail_mode is None):
        ap.error("--fail-after and --fail-mode go together")
    node = Node(a.node_id, a.motor_id, a.data, a.load, store, a.realtime, a.speed, a.broker,
                a.heartbeat, a.fail_after, a.fail_mode, a.flush_wait)

    def on_stop(*_):          # Ctrl+C: finish cleanly, keep what is already saved
        node.stop = True
    signal.signal(signal.SIGINT, on_stop)
    node.run(a.cycles)


if __name__ == "__main__":
    main()
