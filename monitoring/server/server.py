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

3. Recovers missing messages (step 5): every 20 s, for each online node
   with gaps in its message numbers, it asks the node to send them again
   (control/<node_id>/resend). Numbers the node no longer has are recorded
   as "lost for good" and not asked for again. It also reports a "duplicate
   node id" when two programs report under one node name.

4. Answers questions over HTTP (a web API), as JSON:
      GET /motors                    every motor: status (HEALTHY / FAULTY / CHECKING / UNKNOWN),
                                     node state, latest run and latest verdict
      GET /motors/{motor_id}         one motor
      GET /motors/{motor_id}/history recent messages of one motor (?type=verdict&limit=20)
      GET /nodes                     every node: messages received, duplicates ignored,
                                     missing and lost message numbers
      GET /events                    node state changes, newest first (?node_id=node-1&limit=20)
      GET /health                    is the server up and connected to the broker?
   Try them at http://localhost:8000/docs.

5. Shows the dashboard (step 6) at http://localhost:8000: one web page
   (static/dashboard.html) that reads the API above every 2 seconds and
   shows every motor at a glance, the alarms, each motor's verdicts and
   recent runs, the nodes and the events.

6. Security (step 8): the dashboard and API use HTTPS and need a sign-in
   (auth.py); the connection to the broker uses TLS and the "server" login.
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from central_store import CentralStore, now_utc   # noqa: E402
import auth as A                                    # noqa: E402
from messaging import TOPIC_ROOT, CONTROL_ROOT, parse_broker, add_security_args, security_from_args, Security    # noqa: E402

REQUIRED = ("msg_id", "node_id", "motor_id", "seq", "type", "created_at", "payload")


class Receiver:
    """MQTT side: subscribe, validate, store."""

    def __init__(self, store, broker, client_id="monitor-server", timeout=15.0, security=None):
        self.store = store
        self.timeout = timeout
        self.security = security or Security()
        self.host, self.port = parse_broker(broker, self.security.default_port)
        self._last_problem = None
        self.connected = threading.Event()
        self.closing = False
        self.stats = {"new": 0, "duplicate": 0, "rejected": 0}
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=False)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.reconnect_delay_set(1, 10)
        self.client.on_connect_fail = self.on_connect_fail
        self.security.apply(self.client)            # TLS + login (step 8)

    def problem(self, text):
        if text != self._last_problem:              # each new problem once, not on every retry
            self._last_problem = text
            print(f"[server] {text}", flush=True)

    def on_connect_fail(self, client, userdata):
        self.problem(f"can't connect to the broker at {self.host}:{self.port}: "
                     f"{self.security.diagnose(self.host, self.port)}; retrying")

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
            for node_id, epoch, seqs in self.store.requests_due():
                self.client.publish(f"{CONTROL_ROOT}/{node_id}/resend", json.dumps({"type": "resend_request", "epoch": epoch, "seqs": seqs}), qos=1)
                print(f"[server] asked {node_id} to resend {len(seqs)} missing message(s): {_short(seqs)}", flush=True)

    @staticmethod
    def report(evs):
        for ev in (evs if isinstance(evs, list) else [evs]):
            if not ev:
                continue
            effect = {"went silent": f" -> {ev['motor_id']} status UNKNOWN",
                      "connection lost": f" -> {ev['motor_id']} status UNKNOWN",
                      "stopped": f" -> {ev['motor_id']} status UNKNOWN",
                      "online": f" -> {ev['motor_id']} status live again"}.get(ev["event"], "")
            print(f"[server] EVENT {ev['node_id']} ({ev['motor_id']}): {ev['event'].upper()}: {ev['detail']}{effect}",
                  flush=True)

    def stop(self):
        self.closing = True
        self.client.disconnect()
        self.client.loop_stop()

    def on_connect(self, client, userdata, flags, rc, props):
        if rc.is_failure:
            self.problem(f"the broker refused the login as {self.security.user!r}: {rc}; retrying")
            return
        self._last_problem = None
        client.subscribe(f"{TOPIC_ROOT}/#", qos=1)
        self.store.grace()      # we heard nothing while disconnected: don't blame the nodes for it
        self.connected.set()
        resumed = " (resumed session: messages queued while away will follow)" if flags.session_present else ""
        print(f"[server] connected to broker {self.host}:{self.port} ({self.security.describe()}){resumed}", flush=True)

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
            if m.get("type") == "resend_reply":
                self.report(self.store.mark_lost(m))
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


def _short(seqs):
    """[1,2,3,7] -> '#1-#3, #7'"""
    out, i = [], 0
    while i < len(seqs):
        j = i
        while j + 1 < len(seqs) and seqs[j + 1] == seqs[j] + 1:
            j += 1
        out.append(f"#{seqs[i]}" if i == j else f"#{seqs[i]}-#{seqs[j]}")
        i = j + 1
    return ", ".join(out)


def make_app(store, receiver, auth=None):
    app = FastAPI(title="Motor monitoring server",
                  description="Central server of the motor-fault monitoring system. The dashboard is at / (step 6); "
                              "these are the API endpoints it reads. Sign in first (step 8).")
    OPEN = {"/login"}        # the only page you can open without signing in

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """step 8: every page and API call needs a signed-in user (except the sign-in page)"""
        user = None
        if auth and request.url.path not in OPEN:
            user = auth.user_for(request)
            if user is None:
                wants_page = request.method == "GET" and "text/html" in request.headers.get("accept", "")
                if wants_page:
                    response = RedirectResponse(A.login_redirect_target(request), status_code=303)
                else:
                    response = JSONResponse({"detail": "sign in required"}, status_code=401)
                response.headers.update(A.SECURITY_HEADERS)
                return response
        request.state.user = user
        response = await call_next(request)
        response.headers.update(A.SECURITY_HEADERS)
        if request.url.path in ("/", "/login"):
            response.headers["Content-Security-Policy"] = A.PAGE_CSP
        return response

    @app.get("/login", include_in_schema=False)
    def login_form(request: Request, next: str = "/"):
        if auth is None:
            return RedirectResponse("/", status_code=303)
        if auth.user_for(request):
            return RedirectResponse(A.safe_next(next), status_code=303)
        return HTMLResponse(A.login_page(next_path=next), headers={"Cache-Control": "no-store"})

    @app.post("/login", include_in_schema=False)
    async def login(request: Request):
        if auth is None:
            return RedirectResponse("/", status_code=303)
        form = A.parse_form(await request.body())
        user, password, nxt = form.get("user", "").strip(), form.get("password", ""), form.get("next", "/")
        ok, message = auth.check(user, password, A.client_address(request))
        if not ok:
            print(f"[server] failed sign-in as {user!r} from {A.client_address(request)}", flush=True)
            return HTMLResponse(A.login_page(message, nxt, user), status_code=401, headers={"Cache-Control": "no-store"})
        token = auth.new_session(user)
        print(f"[server] {user} signed in from {A.client_address(request)}", flush=True)
        response = RedirectResponse(A.safe_next(nxt), status_code=303)
        response.headers["Set-Cookie"] = auth.cookie_header(token)
        return response

    @app.post("/logout", include_in_schema=False)
    def logout(request: Request):
        response = RedirectResponse("/login", status_code=303)
        if auth:
            auth.end_session(request.cookies.get(A.COOKIE, ""))
            response.headers["Set-Cookie"] = auth.cookie_header("", max_age=0)
        return response

    @app.get("/me")
    def me(request: Request):
        """who is signed in (the dashboard shows it next to the Sign out button)"""
        return {"user": request.state.user, "login_required": auth is not None}

    @app.get("/", include_in_schema=False)
    def dashboard():
        # the dashboard (step 6): one page that reads the API below every 2 s
        return FileResponse(os.path.join(HERE, "static", "dashboard.html"), headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health():
        return {"server": "up", "broker_connected": receiver.connected.is_set(), "heartbeat_timeout_s": receiver.timeout,
                "broker": f"{receiver.host}:{receiver.port}", "messages": receiver.stats, "time": now_utc(),
                "time_ms": int(time.time() * 1000)}   # the dashboard uses this to correct for clock differences

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
    ap.add_argument("--broker", default="localhost", help="host or host:port (default localhost:8883 with TLS)")
    ap.add_argument("--port", type=int, default=8000, help="dashboard and API port (default 8000)")
    ap.add_argument("--db", default=os.path.join(HERE, "central.db"), help="central database file")
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="seconds without a heartbeat before a node counts as offline (default 15)")
    g = add_security_args(ap)     # --ca, --user (default: server), --password-file, --no-tls
    secrets_dir = os.path.join(os.path.dirname(HERE), "secrets")
    g.add_argument("--web-cert", default=os.path.join(secrets_dir, "web.crt"), help="HTTPS certificate for the dashboard")
    g.add_argument("--web-key", default=os.path.join(secrets_dir, "web.key"), help="its private key")
    g.add_argument("--users", default=os.path.join(secrets_dir, "dashboard_users.txt"), help="dashboard logins")
    g.add_argument("--no-login", action="store_true", help="dashboard without sign-in (only on a computer nobody else uses)")
    a = ap.parse_args()
    security = security_from_args(a, default_user="server")
    https = not a.no_tls
    if https and not (os.path.exists(a.web_cert) and os.path.exists(a.web_key)):
        sys.exit(f"No HTTPS certificate ({a.web_cert}). Make the secrets first:  "
                 f"python monitoring/security/setup_security.py   (or use --no-tls)")
    auth = None
    if not a.no_login:
        if not os.path.exists(a.users):
            sys.exit(f"No dashboard users ({a.users}). Make them with:  python monitoring/security/setup_security.py"
                     f"   (or use --no-login)")
        auth = A.Auth(a.users, secure_cookie=https)
    store = CentralStore(a.db)
    receiver = Receiver(store, a.broker, timeout=a.timeout, security=security)
    receiver.start()
    print(f"[server] database {a.db}", flush=True)
    print(f"[server] a node is offline after {a.timeout:g} s without a heartbeat", flush=True)
    scheme = "https" if https else "http"
    print(f"[server] dashboard on {scheme}://localhost:{a.port}   (API: {scheme}://localhost:{a.port}/docs)"
          f"{'' if auth else '   WARNING: no sign-in'}{'' if https else '   WARNING: not encrypted'}", flush=True)
    tls = {"ssl_certfile": a.web_cert, "ssl_keyfile": a.web_key} if https else {}
    try:
        uvicorn.run(make_app(store, receiver, auth), host="0.0.0.0", port=a.port, log_level="warning", **tls)
    finally:
        receiver.stop()
        print("[server] stopped", flush=True)


if __name__ == "__main__":
    main()
