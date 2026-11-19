"""
status.py
==========
Asks the central server for every motor's status and prints a table.

    python monitoring/status.py                     # server at localhost:8000
    python monitoring/status.py --nodes --events    # also message counts per node and recent events
    python monitoring/status.py --watch 2           # refresh every 2 seconds (Ctrl+C to stop)
    python monitoring/status.py --server http://192.168.1.20:8000

Status per motor
  HEALTHY / FAULTY   its node is online; from the latest verdict (or the
                     latest run, before the first full cycle)
  CHECKING           its node is online, no verdict yet
  UNKNOWN            its node is offline or stopped: the last known verdict
                     is shown with its time, but it is not the current status
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def get(server, path):
    with urllib.request.urlopen(server.rstrip("/") + path, timeout=5) as r:
        return json.loads(r.read())


def short_time(iso):
    return iso[11:19] if iso else "-"


def show(a):
    health = get(a.server, "/health")
    motors = get(a.server, "/motors")
    print(f"server up, broker {'connected' if health['broker_connected'] else 'NOT connected'}, "
          f"node offline after {health.get('heartbeat_timeout_s', '?')} s without a heartbeat   (times are UTC)\n")
    if not motors:
        print("no motors have reported yet")
        return
    print(f"{'motor':10s} {'STATUS':9s} {'node':8s} {'node state':24s} {'node is doing':32s} last verdict")
    for m in motors:
        act = m.get("node_activity") or {}
        doing = (f"{act.get('state', '')} ({act.get('runs_done', 0)}/{act.get('runs_per_cycle', '?')})"
                 if m["node_state"] == "online" and act else "-")
        age = m.get("heartbeat_age_s")
        state = m["node_state"] + (f", beat {age}s ago" if age is not None and m["node_state"] == "online" else "")
        v = m["latest_verdict"]
        ver = f"{v['combined']['verdict']} ({v['combined']['confidence']}) at {short_time(v['at'])}" if v else "-"
        print(f"{m['motor_id']:10s} {m['status']:9s} {m['node_id']:8s} {state:24s} {doing[:32]:32s} {ver}")
    unknown = [m for m in motors if m["status"] == "UNKNOWN"]
    for m in unknown:
        print(f"  ! {m['motor_id']}: {m['status_detail']}")
    if a.nodes:
        print(f"\n{'node':8s} {'state':9s} {'received':>8s} {'dupl.':>6s} {'highest #':>9s} {'unsent':>7s}  "
              f"{'missing (asked for)':22s} lost for good")
        for n in get(a.server, "/nodes"):
            uns = (n.get("activity") or {}).get("unsent")
            print(f"{n['node_id']:8s} {n['state'][:9]:9s} {n['received']:8d} {n['duplicates_ignored']:6d} "
                  f"{n['highest_seq']:9d} {('-' if uns is None else uns):>7}  {(n['missing_seq'] or '-'):22s} "
                  f"{n.get('lost_seq') or '-'}")
    if a.events:
        print("\nrecent events (newest first)")
        for e in get(a.server, "/events?limit=10"):
            print(f"  {short_time(e['at'])}  {e['node_id']:8s} {e['motor_id'] or '':10s} {e['event']:16s} {e['detail']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--nodes", action="store_true", help="also show message counts and gaps per node")
    ap.add_argument("--events", action="store_true", help="also show the latest node events")
    ap.add_argument("--watch", type=float, metavar="SECONDS", help="refresh every SECONDS until Ctrl+C")
    a = ap.parse_args()
    try:
        while True:
            if a.watch:
                os.system("cls" if os.name == "nt" else "clear")
                print(f"{time.strftime('%H:%M:%S')}  (refreshing every {a.watch:g} s, Ctrl+C to stop)")
            show(a)
            if not a.watch:
                break
            time.sleep(a.watch)
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"can't reach the server at {a.server} ({e}). Is it running?  python monitoring/server/server.py")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
