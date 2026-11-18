"""
status.py
==========
Asks the central server for every motor's status and prints a table.

    python monitoring/status.py                     # server at localhost:8000
    python monitoring/status.py --server http://192.168.1.20:8000 --nodes
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def get(server, path):
    with urllib.request.urlopen(server.rstrip("/") + path, timeout=5) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--nodes", action="store_true", help="also show message counts and gaps per node")
    a = ap.parse_args()
    try:
        health = get(a.server, "/health")
        motors = get(a.server, "/motors")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"can't reach the server at {a.server} ({e}). Is it running?  python monitoring/server/server.py")
    print(f"server up, broker {'connected' if health['broker_connected'] else 'NOT connected'}; "
          f"since the server started: {health['messages']['new']} messages stored, "
          f"{health['messages']['duplicate']} duplicates ignored\n")
    if not motors:
        print("no motors have reported yet")
    print(f"{'motor':10s} {'node':8s} {'load':9s} {'last heard (UTC)':26s} {'latest run':28s} verdict (combined)")
    for m in motors:
        r, v = m["latest_run"], m["latest_verdict"]
        run = f"{r['run']} {'FAULTY' if r['faulty'] else 'healthy'}" if r else "-"
        ver = (f"{v['combined']['verdict']} ({v['combined']['confidence']})"
               f"{'' if v['agree'] else '  triggers/ML disagree'}") if v else "- (no full cycle yet)"
        print(f"{m['motor_id']:10s} {m['node_id']:8s} {('loaded' if m['load'] else 'unloaded'):9s} "
              f"{m['last_seen']:26s} {run:28s} {ver}")
    if a.nodes:
        print(f"\n{'node':8s} {'received':>8s} {'dupl.':>6s} {'highest #':>9s}  missing")
        for n in get(a.server, "/nodes"):
            print(f"{n['node_id']:8s} {n['received']:8d} {n['duplicates_ignored']:6d} {n['highest_seq']:9d}  "
                  f"{n['missing_seq'] or '-'}")


if __name__ == "__main__":
    main()
