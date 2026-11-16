"""
node.py
========
A simulated monitoring node: a program that pretends to be the small
computer attached to one motor.

    python monitoring/node.py --node-id node-1 --motor-id pump-A --data Loaded/RU --load 1
    python monitoring/node.py --node-id node-2 --motor-id fan-B --data Unloaded/HH --load 0 --realtime --speed 5

What it does
  1. "Sensors": replays the motor's CSV recordings one second at a time,
     the way live sensors would deliver them (SensorReplay). With
     --realtime it also waits, so a 10 s run takes 10 s (or 10/speed s).
  2. After each run: works out the features and the Stage A check
     (is this run above the healthy level?) and saves a "run_result".
  3. After a full cycle of runs (normally all 8 speed profiles): gives the
     three verdicts (triggers, ML, combined) and saves a "verdict".
  4. Everything is saved in the node's local store (local_store.py), as
     the messages it will later send over the network.

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
"""
import argparse
import datetime as dt
import glob
import os
import re
import signal
import sys
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
from local_store import LocalStore            # noqa: E402

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
    def __init__(self, node_id, motor_id, data_dir, load, store_path, realtime=False, speed=1.0):
        self.node_id, self.motor_id, self.load = node_id, motor_id, load
        self.realtime, self.speed = realtime, speed
        self.files = self._find_runs(data_dir, load)
        self.store = LocalStore(store_path, node_id, motor_id)
        self.seq = self.store.next_seq()
        self.healthy = PM.read_calibration(
            os.path.join(ROOT, f"trigger_calibration_final_load{load}.json"))["healthy_reference"]
        self.stop = False

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
        """build a message and save it locally (sending comes in step 2)"""
        msg = {"msg_id": f"{self.node_id}-{self.seq}", "seq": self.seq, "type": type_,
               "node_id": self.node_id, "motor_id": self.motor_id, "load": self.load,
               "created_at": now_utc(), "payload": jsonable(payload)}
        self.store.add(msg)
        self.seq += 1
        return msg

    def record_run(self, path):
        """receive one run from the sensors, block by block; None if stopped"""
        blocks = []
        for b in SensorReplay(path, realtime=self.realtime, speed=self.speed).blocks():
            if self.stop:
                return None
            blocks.append(b)
        df = pd.concat(blocks, ignore_index=True)
        df.insert(0, "Time", np.arange(len(df)) / FALLBACK_FS)
        return df

    def run_cycle(self, cycle):
        vectors = {}
        for path in self.files:
            cid = PM.speed_of(path, None)
            df = self.record_run(path)
            if df is None:
                return False
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

        files = list(vectors)
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
        self.log(f"cycle {cycle} done: triggers {hand['verdict']} ({hand['confidence']}), "
                 f"ML {ml['verdict']} ({ml['confidence']}), combined {verdict} ({conf})")
        return True

    def run(self, cycles=1):
        self.log(f"start: {len(self.files)} runs per cycle, load {self.load}, "
                 f"{'real time x' + str(self.speed) if self.realtime else 'fast replay'}, next seq {self.seq}")
        c = 1
        while not self.stop and (cycles == 0 or c <= cycles):
            if not self.run_cycle(c):
                break
            c += 1
        self.log("stopped" if self.stop else "finished")
        self.store.close()


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
    a = ap.parse_args()
    store = a.store or os.path.join(HERE, "stores", f"{a.node_id}.db")
    node = Node(a.node_id, a.motor_id, a.data, a.load, store, a.realtime, a.speed)

    def on_stop(*_):          # Ctrl+C: finish cleanly, keep what is already saved
        node.stop = True
    signal.signal(signal.SIGINT, on_stop)
    node.run(a.cycles)


if __name__ == "__main__":
    main()
