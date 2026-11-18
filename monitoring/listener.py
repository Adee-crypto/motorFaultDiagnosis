"""
listener.py
============
Shows every message that arrives at the broker, from all nodes (step 2).
It stands in for the central server until step 3.

    python monitoring/listener.py                    # broker on localhost:1883
    python monitoring/listener.py --broker 192.168.1.20:1883 --json

It subscribes to  monitor/#  (every node, every message type) and prints
one line per message. It also counts messages per node and points out a
gap or a repeat in a node's message numbers (seq), which is the first sign
of a lost or doubled message.
"""
import argparse
import datetime as dt
import json
import os
import sys

import paho.mqtt.client as mqtt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from messaging import TOPIC_ROOT, parse_broker  # noqa: E402


class Listener:
    def __init__(self, show_json=False, show_heartbeats=False):
        self.show_json = show_json
        self.show_heartbeats = show_heartbeats
        self.last_seq = {}
        self.count = {}

    def line(self, m):
        p = m["payload"]
        head = f"{m['node_id']} ({m['motor_id']}) #{m['seq']}"
        if m["type"] == "run_result":
            return (f"{head:28s} run {p['run']:12s} [{p['speed']:19s}] "
                    f"{'FAULTY' if p['faulty'] else 'healthy level':13s} "
                    f"accel1 {p['accel1_x_healthy']:.1f}x  mic {p['mic_x_healthy']:.1f}x")
        if m["type"] == "verdict":
            return (f"{head:28s} VERDICT cycle {p['cycle']}: triggers {p['triggers']['verdict']} "
                    f"({p['triggers']['confidence']}) | ML {p['ml']['verdict']} ({p['ml']['confidence']}) | "
                    f"combined {p['combined']['verdict']} ({p['combined']['confidence']})"
                    f"{'' if p['agree'] else '  <- triggers and ML DISAGREE'}")
        return f"{head:28s} {m['type']}"

    def on_message(self, client, userdata, msg):
        try:
            m = json.loads(msg.payload)
        except ValueError:
            print(f"[{dt.datetime.now():%H:%M:%S}] unreadable message on {msg.topic}")
            return
        t = f"[{dt.datetime.now():%H:%M:%S}] "
        if m.get("type") == "heartbeat":
            if self.show_heartbeats:
                print(t + (json.dumps(m) if self.show_json else
                           f"{m['node_id']} ({m.get('motor_id')}) heartbeat: {m.get('state')} "
                           f"({m.get('runs_done')}/{m.get('runs_per_cycle')}), unsent {m.get('unsent')}"), flush=True)
            return
        if m.get("type") == "status":
            print(t + f"{m['node_id']} ({m.get('motor_id')}) is {m['status'].upper()}: {m.get('reason')}"
                  f"{'  (kept by the broker from earlier)' if msg.retain else ''}", flush=True)
            return
        node, seq = m.get("node_id"), m.get("seq")
        note = ""
        last = self.last_seq.get(node)
        if last is None and seq is not None and seq > 1:
            note = f"  (first message seen from this node; #1-#{seq - 1} not received here)"
        if last is not None and seq is not None:
            if seq <= last:
                note = f"  (repeat or out of order: already had up to #{last})"
            elif seq > last + 1:
                note = f"  (GAP: #{last + 1}-#{seq - 1} missing)"
        if seq is not None and (last is None or seq > last):
            self.last_seq[node] = seq
        self.count[node] = self.count.get(node, 0) + 1
        print(f"[{dt.datetime.now():%H:%M:%S}] " + (json.dumps(m) if self.show_json else self.line(m)) + note,
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broker", default="localhost", help="host or host:port (default localhost:1883)")
    ap.add_argument("--json", action="store_true", help="print the raw messages")
    ap.add_argument("--heartbeats", action="store_true", help="also print heartbeats (every few seconds per node)")
    a = ap.parse_args()
    host, port = parse_broker(a.broker)
    lis = Listener(a.json, a.heartbeats)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="listener")

    def on_connect(c, u, f, rc, p):
        if rc.is_failure:
            print(f"broker refused the connection: {rc}")
            return
        c.subscribe(f"{TOPIC_ROOT}/#", qos=1)
        print(f"listening on {host}:{port}, topic {TOPIC_ROOT}/#  (Ctrl+C to stop)", flush=True)

    client.on_connect = on_connect
    client.on_message = lis.on_message
    try:
        client.connect(host, port, keepalive=30)
    except OSError as e:
        sys.exit(f"can't reach the broker at {host}:{port} ({e}). Is it running?  "
                 f"docker compose -f monitoring/docker-compose.yml up -d")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nmessages received per node: " + (", ".join(f"{k} {v}" for k, v in lis.count.items()) or "none"))


if __name__ == "__main__":
    main()
