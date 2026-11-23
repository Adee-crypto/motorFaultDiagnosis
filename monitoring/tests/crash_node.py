"""
TEST (step 7): make the node program inside a container fail, to watch the
system react. Run it inside a node's container:

    docker compose exec node-3 python monitoring/tests/crash_node.py           # crash
    docker compose exec node-3 python monitoring/tests/crash_node.py --hang 30 # freeze 30 s

crash : kills the node program at once (no goodbye). The broker announces its
        last will, the dashboard shows the motor as UNKNOWN, and Docker starts the
        node again by itself (restart: unless-stopped). The new node sends what
        it had kept and carries on numbering from its store.
hang  : freezes the node program (it stops sending heartbeats) for N seconds,
        then lets it continue. After 15 s the server marks it offline; when it
        wakes up it comes back online by itself.
"""
import argparse
import os
import signal
import time


def node_pids():
    me = os.getpid()
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as fh:
                cmd = fh.read().split(b"\0")
        except OSError:
            continue
        if os.path.basename(cmd[0]).startswith(b"python") and any(part.endswith(b"node.py") for part in cmd):
            pids.append(int(d))
    return pids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hang", type=float, metavar="SECONDS", help="freeze instead of crashing")
    a = ap.parse_args()
    pids = node_pids()
    if not pids:
        raise SystemExit("no node program found in this container")
    for pid in pids:
        if pid == 1:
            raise SystemExit("the node is process 1 here; this test needs 'init: true' in docker-compose.yml")
    if a.hang:
        for pid in pids:
            os.kill(pid, signal.SIGSTOP)
        print(f"node frozen (pid {', '.join(map(str, pids))}); it continues in {a.hang:g} s", flush=True)
        time.sleep(a.hang)
        for pid in pids:
            os.kill(pid, signal.SIGCONT)
        print("node continues", flush=True)
    else:
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
        print(f"node crashed (pid {', '.join(map(str, pids))}); Docker will start it again", flush=True)


if __name__ == "__main__":
    main()
