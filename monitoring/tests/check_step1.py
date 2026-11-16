"""
check_step1.py
===============
Step 1 check: a simulated node must give exactly the same results as
predict_all.py on the same recordings.

    python monitoring/tests/check_step1.py <Unloaded folder> <Loaded folder>

For every motor folder (BR, FB, HH, KA, RM, RU, SW, VU) and both loads it
runs one node cycle into a temporary store and compares the saved verdict
(three verdicts and all points) with predict_all.diagnose_all.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MON = os.path.dirname(HERE)
ROOT = os.path.dirname(MON)
for p in (ROOT, os.path.join(ROOT, "ml"), MON):
    sys.path.insert(0, p)
import predict_all as PA   # noqa: E402
from node import Node      # noqa: E402

MOTORS = ["HH", "BR", "FB", "KA", "RM", "RU", "SW", "VU"]


def same_points(a, b):
    keys = set(a) | set(b)
    return all(abs(a.get(k, 0) - b.get(k, 0)) < 1e-3 for k in keys)


def main():
    folders = {0: sys.argv[1], 1: sys.argv[2]}
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for load, root in folders.items():
            for m in MOTORS:
                d = os.path.join(root, m)
                node = Node(f"test-{load}-{m}", m, d, load, os.path.join(tmp, f"{load}-{m}.db"))
                node.log = lambda *_: None
                node.run(1)
                from local_store import LocalStore
                got = LocalStore(os.path.join(tmp, f"{load}-{m}.db"), f"test-{load}-{m}", m).latest(1, "verdict")[0]["payload"]
                ref = PA.diagnose_all(node.files, load, verbose=False)
                ok = all(got[k]["verdict"] == ref[k]["verdict"] and got[k]["confidence"] == ref[k]["confidence"]
                         and same_points(got[k]["points"], ref[k]["points"]) for k in ("triggers", "ml", "combined"))
                ok &= got["faulty_runs"] == ref["faulty_runs"]
                bad += not ok
                print(f"load {load} {m}: {'OK  ' if ok else 'DIFF'} node -> {got['combined']['verdict']} "
                      f"({got['combined']['confidence']}), predict_all -> {ref['combined']['verdict']}", flush=True)
    print("ALL MATCH" if bad == 0 else f"{bad} MISMATCHES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
