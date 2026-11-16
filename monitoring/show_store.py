"""
show_store.py
==============
Prints what a node has saved in its local store.

    python monitoring/show_store.py monitoring/stores/node-1.db [--last 20] [--verdicts]
"""
import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_store import LocalStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--verdicts", action="store_true", help="only the verdicts")
    a = ap.parse_args()
    if not os.path.exists(a.store):
        sys.exit(f"no store at {a.store}")
    meta = dict(sqlite3.connect(a.store).execute("SELECT key, value FROM meta").fetchall())
    st = LocalStore(a.store, meta["node_id"], meta["motor_id"])
    msgs = st.latest(a.last, "verdict" if a.verdicts else None)
    print(f"{meta['node_id']} ({meta['motor_id']}): showing {len(msgs)} message(s), "
          f"{len(st.unsent())} not yet sent")
    for m in msgs:
        p = m["payload"]
        head = f"#{m['seq']:<4} {m['created_at']}  {m['type']:10s}"
        if m["type"] == "run_result":
            print(f"{head} {p['run']:12s} {'FAULTY' if p['faulty'] else 'healthy':8s} "
                  f"accel1 {p['accel1_x_healthy']}x  mic {p['mic_x_healthy']}x")
        else:
            print(f"{head} cycle {p['cycle']}: triggers {p['triggers']['verdict']} ({p['triggers']['confidence']}) | "
                  f"ML {p['ml']['verdict']} ({p['ml']['confidence']}) | "
                  f"combined {p['combined']['verdict']} ({p['combined']['confidence']})"
                  f"{'' if p['agree'] else '  <- DISAGREE'}")


if __name__ == "__main__":
    main()
