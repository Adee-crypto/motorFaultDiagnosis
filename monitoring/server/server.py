"""
server.py
==========
The central server (step 3).

    python monitoring/server/server.py                       # broker localhost:1883, web on port 8000
    python monitoring/server/server.py --broker localhost:1883 --port 8000 --db monitoring/server/central.db --timeout 15

Two jobs in one program:

1. Receives every message from every node (MQTT subscriber on monitor/#)
   and saves it in the central database (central_store.py). A message that
   arrives twice is saved once. It uses a lasting broker session
   (client id "monitor-server", clean_session off): while the server is
   down, the broker keeps its messages and hands them over when the server
   comes back.

2. Watches every node (step 4): a node is "online" while its heartbeats
   arrive. No heartbeat for --timeout seconds (default 15), or the broker
   announcing its "last will", makes it "offline"; a goodbye makes it
   "stopped". A motor whose node is not online gets status UNKNOWN.
   Every change is logged as an event.

3. Answers questions over HTTP (a web API), as JSON:
      GET /motors                    every motor: status (HEALTHY / FAULTY / CHECKING / UNKNOWN),
                                     node state, latest run and latest verdict
      GET /motors/{motor_id}         one motor
      GET /motors/{motor_id}/history recent messages of one motor (?type=verdict&limit=20)
      GET /nodes                     every node: messages received, duplicates ignored,
                                     missing message numbers
      GET /events                    node state changes, newest first (?node_id=node-1&limit=20)
      GET /health                    is the server up and connected to the broker?
   Open http://localhost:8000 (or /docs) in a browser to try them.
"""
import argparse
import json
import os
import sys
import threading
import time
from typing import Optional

import paho.mqtt.client as mqtt
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from central_store import CentralStore, now_utc   # noqa: E402
from messaging import TOPIC_ROOT, parse_broker    # noqa: E402

REQUIRED = ("msg_id", "node_id", "motor_id", "seq", "type", "created_at", "payload")


class Receiver:
    """MQTT side: subscribe, validate, store."""

    def __init__(self, store, broker, client_id="monitor-server", timeout=15.0):
        self.store = store
        self.timeout = timeout
        self.host, self.port = parse_broker(broker)
        self.connected = threading.Event()
        self.closing = False
        self.stats = {"new": 0, "duplicate": 0, "rejected": 0}
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=False)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.reconnect_delay_set(1, 10)

    def start(self):
        self.client.connect_async(self.host, self.port, keepalive=30)
        self.client.loop_start()
        threading.Thread(target=self.watchdog, daemon=True).start()

    def watchdog(self):
        """every second: online nodes that went quiet for longer than the timeout become offline"""
        while not self.closing:
            time.sleep(1)
            if not self.connected.is_set():
                continue        # no broker: we can't hear anyone, so don't blame the nodes
            for ev in self.store.check_timeouts(self.timeout):
                self.report(ev)

    @staticmethod
    def report(ev):
        if not ev:
            return
        effect = "status UNKNOWN" if ev["event"] in ("went silent", "connection lost", "stopped") else "status live again"
        print(f"[server] EVENT {ev['node_id']} ({ev['motor_id']}): {ev['event'].upper()}: {ev['detail']} "
              f"-> {ev['motor_id']} {effect}", flush=True)

    def stop(self):
        self.closing = True
        self.client.disconnect()
        self.client.loop_stop()

    def on_connect(self, client, userdata, flags, rc, props):
        if rc.is_failure:
            print(f"broker refused the connection: {rc}", flush=True)
            return
        client.subscribe(f"{TOPIC_ROOT}/#", qos=1)
        self.store.grace()      # we heard nothing while disconnected: don't blame the nodes for it
        self.connected.set()
        resumed = " (resumed session: messages queued while away will follow)" if flags.session_present else ""
        print(f"[server] connected to broker {self.host}:{self.port}{resumed}", flush=True)

    def on_disconnect(self, client, userdata, flags, rc, props):
        if self.connected.is_set() and not self.closing:
            print(f"[server] lost connection to broker ({rc}); reconnecting", flush=True)
        self.connected.clear()

    def on_message(self, client, userdata, msg):
        try:
            m = json.loads(msg.payload)
            if m.get("type") == "heartbeat":
                self.report(self.store.heartbeat(m))
                return
            if m.get("type") == "status":
                self.report(self.store.node_status(m))
                return
            if not all(k in m for k in REQUIRED):
                raise ValueError("missing fields")
        except ValueError as e:
            self.stats["rejected"] += 1
            print(f"[server] rejected a message on {msg.topic}: {e}", flush=True)
            return
        result = self.store.add(m)
        self.stats[result] += 1
        if self.store.restart_event:
            ev = self.store.restart_event
            print(f"[server] EVENT {ev['node_id']} ({ev['motor_id']}): NUMBERING RESTARTED: {ev['detail']}", flush=True)
        p = m["payload"]
        what = (f"run {p.get('run')} {'FAULTY' if p.get('faulty') else 'healthy level'}" if m["type"] == "run_result"
                else f"VERDICT {p['combined']['verdict']} ({p['combined']['confidence']})" if m["type"] == "verdict"
                else m["type"])
        print(f"[server] {m['node_id']} #{m['seq']:<4} {m['motor_id']:10s} {what}"
              f"{'  (duplicate, ignored)' if result == 'duplicate' else ''}", flush=True)


def make_app(store, receiver):
    app = FastAPI(title="Motor monitoring server",
                  description="Central server of the motor-fault monitoring system (step 3).")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health")
    def health():
        return {"server": "up", "broker_connected": receiver.connected.is_set(), "heartbeat_timeout_s": receiver.timeout,
                "broker": f"{receiver.host}:{receiver.port}", "messages": receiver.stats, "time": now_utc()}

    @app.get("/motors")
    def motors():
        return store.motors()

    @app.get("/motors/{motor_id}")
    def motor(motor_id: str):
        m = store.motor(motor_id)
        if m is None:
            raise HTTPException(404, f"no motor {motor_id!r}")
        return m

    @app.get("/motors/{motor_id}/history")
    def history(motor_id: str, type: Optional[str] = None, limit: int = 50):
        if store.motor(motor_id) is None:
            raise HTTPException(404, f"no motor {motor_id!r}")
        return store.history(motor_id, type, min(limit, 1000))

    @app.get("/nodes")
    def nodes():
        return store.nodes()

    @app.get("/events")
    def events(node_id: Optional[str] = None, limit: int = 50):
        return store.events(min(limit, 1000), node_id)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broker", default="localhost", help="host or host:port (default localhost:1883)")
    ap.add_argument("--port", type=int, default=8000, help="web API port (default 8000)")
    ap.add_argument("--db", default=os.path.join(HERE, "central.db"), help="central database file")
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="seconds without a heartbeat before a node counts as offline (default 15)")
    a = ap.parse_args()
    store = CentralStore(a.db)
    receiver = Receiver(store, a.broker, timeout=a.timeout)
    receiver.start()
    print(f"[server] database {a.db}", flush=True)
    print(f"[server] a node is offline after {a.timeout:g} s without a heartbeat", flush=True)
    print(f"[server] web API on http://localhost:{a.port}  (try http://localhost:{a.port}/docs)", flush=True)
    try:
        uvicorn.run(make_app(store, receiver), host="0.0.0.0", port=a.port, log_level="warning")
    finally:
        receiver.stop()
        print("[server] stopped", flush=True)


if __name__ == "__main__":
    main()
