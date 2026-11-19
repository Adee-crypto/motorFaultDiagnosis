"""
central_store.py
=================
The central database (step 3): one SQLite file that holds every message
from every node, and each motor's latest status.

Tables
  messages   every message received, once. The key is msg_id
             ("<node_id>-<epoch>-<seq>"), so a message delivered twice by the
             broker (MQTT "at least once") is stored only once.
  motors     one row per motor: its node, load, when it was last heard
             from, its latest run result and its latest verdict.
  nodes      one row per node: messages received, highest message number,
             which numbers are missing (lost or not yet sent), and (step 4)
             its state: online / offline / stopped, since when and why,
             and its last heartbeat.
  events     (step 4) every change of a node's state, with time and reason.

Node states (step 4)
  online    heartbeats are arriving
  offline   heartbeats stopped (timeout), or the broker announced the node's
            "last will" (connection lost without a goodbye)
  stopped   the node said goodbye (finished or stopped by the user)
A motor whose node is not online has status UNKNOWN: its last result is
still shown, with its time, but never as the current status.

Store-and-forward (step 5)
  missing     message numbers below a node's highest one that never arrived;
              the server asks the node for them (requests_due)
  lost        numbers the node says it no longer has: recorded once, with an
              event, and no longer asked for
  run_id      each run of a node program has its own id. Heartbeats from two
              runs alternating under one node name mean two programs share a
              name: one "duplicate node id" event (at most once a minute).
              A status message from an earlier run (e.g. its late "last will")
              is ignored, so it can't mark the current run offline.

All writes go through one lock, because the MQTT thread writes while the
web interface reads.
"""
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CentralStore:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        self.restart_event = None
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                msg_id TEXT PRIMARY KEY, node_id TEXT, motor_id TEXT, seq INTEGER, type TEXT,
                created_at TEXT, received_at TEXT, body TEXT);
            CREATE INDEX IF NOT EXISTS idx_msg_motor ON messages (motor_id, type, created_at);
            CREATE INDEX IF NOT EXISTS idx_msg_node ON messages (node_id, seq);
            CREATE TABLE IF NOT EXISTS motors (
                motor_id TEXT PRIMARY KEY, node_id TEXT, load INTEGER, last_seen TEXT,
                last_run TEXT, last_verdict TEXT);
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY, received INTEGER DEFAULT 0, duplicates INTEGER DEFAULT 0,
                max_seq INTEGER DEFAULT 0, last_seen TEXT);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, node_id TEXT, motor_id TEXT,
                event TEXT, detail TEXT);
        """)
        # step 4 columns; added to a step-3 database if missing
        have = {r[1] for r in self.db.execute("PRAGMA table_info(nodes)")}
        for col, kind in (("motor_id", "TEXT"), ("state", "TEXT"), ("state_since", "TEXT"), ("reason", "TEXT"),
                          ("last_heartbeat", "TEXT"), ("last_hb_epoch", "REAL"), ("activity", "TEXT"),
                          ("epoch", "TEXT DEFAULT ''"), ("epoch_started", "TEXT"),
                          ("run_id", "TEXT"), ("prev_run_id", "TEXT"), ("dup_reported", "REAL"),
                          ("lost", "TEXT DEFAULT '[]'"), ("last_request", "REAL"), ("epochs", "TEXT DEFAULT '[]'")):
            if col not in have:
                self.db.execute(f"ALTER TABLE nodes ADD COLUMN {col} {kind}")
        if "epoch" not in {r[1] for r in self.db.execute("PRAGMA table_info(messages)")}:
            self.db.execute("ALTER TABLE messages ADD COLUMN epoch TEXT DEFAULT ''")
        # after a server restart, give online nodes one timeout to send a heartbeat
        self.db.execute("UPDATE nodes SET last_hb_epoch = ? WHERE state = 'online'", (time.time(),))
        self.db.commit()

    # ------------------------------------------------------------ writing
    def add(self, m):
        """store one message; returns 'new' or 'duplicate'"""
        epoch = m.get("epoch", "")
        with self.lock:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO messages (msg_id, node_id, motor_id, seq, type, created_at, received_at, body, epoch) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (m["msg_id"], m["node_id"], m["motor_id"], m["seq"], m["type"], m["created_at"],
                 now_utc(), json.dumps(m), epoch))
            new = cur.rowcount == 1
            self._ensure_node(m["node_id"], m["motor_id"])
            self.restart_event = None
            if new:
                node = self.db.execute("SELECT epoch, epoch_started, epochs FROM nodes WHERE node_id = ?",
                                       (m["node_id"],)).fetchone()
                seen = set(json.loads(node["epochs"] or "[]"))
                if epoch not in seen:
                    self.db.execute("UPDATE nodes SET epochs = ? WHERE node_id = ?",
                                    (json.dumps(sorted(seen | {epoch})), m["node_id"]))
                # only a never-seen epoch counts as a restart; an epoch seen before (e.g. a second
                # program wrongly using the same node name) must not flip the numbering back and forth
                if (node["epoch"] or "") != epoch and epoch not in seen and (node["epoch_started"] or "") <= m["created_at"]:
                    # a fresh local store on the node: numbering starts again under a new epoch
                    if node["epoch"]:     # not for a node upgrading from step 3 (no epoch before)
                        ev = {"at": now_utc(), "node_id": m["node_id"], "motor_id": m["motor_id"],
                              "event": "numbering restarted",
                              "detail": f"node has a fresh local store (epoch {epoch}); message numbers start again"}
                        self.db.execute("INSERT INTO events (at, node_id, motor_id, event, detail) VALUES (?, ?, ?, ?, ?)",
                                        tuple(ev.values()))
                        self.restart_event = ev
                    hi = self.db.execute("SELECT MAX(seq) FROM messages WHERE node_id = ? AND epoch = ?",
                                         (m["node_id"], epoch)).fetchone()[0] or 0
                    self.db.execute("UPDATE nodes SET epoch = ?, epoch_started = ?, max_seq = ? WHERE node_id = ?",
                                    (epoch, m["created_at"], hi, m["node_id"]))
                cur_epoch = self.db.execute("SELECT epoch FROM nodes WHERE node_id = ?", (m["node_id"],)).fetchone()[0]
                self.db.execute("UPDATE nodes SET received = received + 1, last_seen = ? WHERE node_id = ?",
                                (now_utc(), m["node_id"]))
                if (cur_epoch or "") == epoch:
                    self.db.execute("UPDATE nodes SET max_seq = MAX(max_seq, ?) WHERE node_id = ?", (m["seq"], m["node_id"]))
                self._update_motor(m)
            else:
                self.db.execute("UPDATE nodes SET duplicates = duplicates + 1 WHERE node_id = ?",
                                (m["node_id"],))
            self.db.commit()
        return "new" if new else "duplicate"

    def _update_motor(self, m):
        self.db.execute("INSERT OR IGNORE INTO motors (motor_id) VALUES (?)", (m["motor_id"],))
        self.db.execute("UPDATE motors SET node_id = ?, load = ?, last_seen = ? WHERE motor_id = ?",
                        (m["node_id"], m["load"], now_utc(), m["motor_id"]))
        col = {"run_result": "last_run", "verdict": "last_verdict"}.get(m["type"])
        if col:
            # a late (backlogged) message must not overwrite a newer one
            row = self.db.execute(f"SELECT {col} FROM motors WHERE motor_id = ?", (m["motor_id"],)).fetchone()
            old = json.loads(row[0]) if row and row[0] else None
            if old is None or (m["created_at"], m["seq"]) >= (old["created_at"], old["seq"]):
                self.db.execute(f"UPDATE motors SET {col} = ? WHERE motor_id = ?", (json.dumps(m), m["motor_id"]))

    # ------------------------------------------------ node liveness (step 4)
    def _set_state(self, node_id, motor_id, state, reason, event, detail):
        row = self.db.execute("SELECT state FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        old = row["state"] if row else None
        if old == state:
            return None
        self.db.execute("UPDATE nodes SET state = ?, state_since = ?, reason = ? WHERE node_id = ?",
                        (state, now_utc(), reason, node_id))
        ev = {"at": now_utc(), "node_id": node_id, "motor_id": motor_id, "event": event, "detail": detail}
        self.db.execute("INSERT INTO events (at, node_id, motor_id, event, detail) VALUES (?, ?, ?, ?, ?)",
                        tuple(ev.values()))
        return ev

    def _ensure_node(self, node_id, motor_id):
        self.db.execute("INSERT OR IGNORE INTO nodes (node_id) VALUES (?)", (node_id,))
        if motor_id:
            self.db.execute("UPDATE nodes SET motor_id = ? WHERE node_id = ?", (motor_id, node_id))

    def _check_run(self, node_id, motor_id, run_id):
        """track which program run speaks for a node; returns a 'duplicate node id' event or None"""
        if not run_id:
            return None
        r = self.db.execute("SELECT run_id, prev_run_id, dup_reported FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if r["run_id"] == run_id:
            return None
        ev = None
        if r["prev_run_id"] == run_id and time.time() - (r["dup_reported"] or 0) > 60:
            # a run we had already seen replaced is talking again: two programs share this name
            ev = {"at": now_utc(), "node_id": node_id, "motor_id": motor_id, "event": "duplicate node id",
                  "detail": f"two programs are reporting as {node_id} (runs {r['run_id']} and {run_id}); "
                            f"give each node its own --node-id"}
            self.db.execute("INSERT INTO events (at, node_id, motor_id, event, detail) VALUES (?, ?, ?, ?, ?)",
                            tuple(ev.values()))
            self.db.execute("UPDATE nodes SET dup_reported = ? WHERE node_id = ?", (time.time(), node_id))
        self.db.execute("UPDATE nodes SET prev_run_id = run_id, run_id = ? WHERE node_id = ?", (run_id, node_id))
        return ev

    def heartbeat(self, hb):
        """a heartbeat arrived: the node is online; returns the events it causes"""
        with self.lock:
            self._ensure_node(hb["node_id"], hb.get("motor_id"))
            dup = self._check_run(hb["node_id"], hb.get("motor_id"), hb.get("run_id"))
            info = {k: hb.get(k) for k in ("state", "cycle", "runs_done", "runs_per_cycle", "unsent", "uptime_s", "interval_s")}
            self.db.execute("UPDATE nodes SET last_heartbeat = ?, last_hb_epoch = ?, activity = ? WHERE node_id = ?",
                            (now_utc(), time.time(), json.dumps(info), hb["node_id"]))
            ev = self._set_state(hb["node_id"], hb.get("motor_id"), "online", "heartbeats arriving",
                                 "online", "heartbeats arriving")
            if ev:
                self._first_request_soon(hb["node_id"])
            self.db.commit()
        return [e for e in (dup, ev) if e]

    def node_status(self, st):
        """a status message: "online" when the node connects, "offline" on goodbye or last will"""
        with self.lock:
            self._ensure_node(st["node_id"], st.get("motor_id"))
            cur = self.db.execute("SELECT run_id FROM nodes WHERE node_id = ?", (st["node_id"],)).fetchone()[0]
            if st["status"] == "offline" and st.get("run_id") and cur and st["run_id"] != cur:
                return None      # an earlier run's goodbye or last will: the current run is unaffected
            if st["status"] == "online":
                self._check_run(st["node_id"], st.get("motor_id"), st.get("run_id"))
                self.db.execute("UPDATE nodes SET last_hb_epoch = ? WHERE node_id = ?", (time.time(), st["node_id"]))
                ev = self._set_state(st["node_id"], st.get("motor_id"), "online", "connected", "online", "connected")
                if ev:
                    self._first_request_soon(st["node_id"])
            elif st.get("reason") == "connection lost":
                ev = self._set_state(st["node_id"], st.get("motor_id"), "offline", "connection lost",
                                     "connection lost", "broker announced the node's last will (no goodbye)")
            else:
                ev = self._set_state(st["node_id"], st.get("motor_id"), "stopped", st.get("reason"),
                                     "stopped", f"node said goodbye: {st.get('reason')}")
            self.db.commit()
        return ev

    # ------------------------------------------------ store-and-forward (step 5)
    REQUEST_EVERY = 20.0      # seconds between resend requests to one node

    def _first_request_soon(self, node_id):
        """a node just came online: give it ~10 s to send its own backlog before asking"""
        self.db.execute("UPDATE nodes SET last_request = ? WHERE node_id = ?",
                        (time.time() - self.REQUEST_EVERY + 10, node_id))

    def _missing(self, r):
        have = {q for (q,) in self.db.execute("SELECT seq FROM messages WHERE node_id = ? AND epoch = ?",
                                              (r["node_id"], r["epoch"] or ""))}
        lost = set(json.loads(r["lost"] or "[]"))
        return [q for q in range(1, (r["max_seq"] or 0) + 1) if q not in have and q not in lost]

    def requests_due(self, limit=200):
        """online nodes with missing messages whose last request is old enough: [(node_id, epoch, seqs)]"""
        out, now = [], time.time()
        with self.lock:
            for r in self.db.execute("SELECT * FROM nodes WHERE state = 'online'").fetchall():
                if now - (r["last_request"] or 0) < self.REQUEST_EVERY:
                    continue
                miss = self._missing(r)
                if miss:
                    out.append((r["node_id"], r["epoch"] or "", miss[:limit]))
                    self.db.execute("UPDATE nodes SET last_request = ? WHERE node_id = ?", (now, r["node_id"]))
            self.db.commit()
        return out

    def mark_lost(self, reply):
        """the node says it no longer has these numbers: record them once, stop asking"""
        with self.lock:
            r = self.db.execute("SELECT * FROM nodes WHERE node_id = ?", (reply["node_id"],)).fetchone()
            if r is None or (reply.get("epoch", "") != (r["epoch"] or "")):
                return None          # about an older store: those numbers are not counted as missing anyway
            lost = set(json.loads(r["lost"] or "[]"))
            new = sorted(set(int(q) for q in reply.get("unavailable", [])) - lost)
            if not new:
                return None
            self.db.execute("UPDATE nodes SET lost = ? WHERE node_id = ?",
                            (json.dumps(sorted(lost | set(new))), reply["node_id"]))
            ev = {"at": now_utc(), "node_id": reply["node_id"], "motor_id": reply.get("motor_id"),
                  "event": "lost for good", "detail": f"messages {_ranges(new)}: {reply.get('reason')}"}
            self.db.execute("INSERT INTO events (at, node_id, motor_id, event, detail) VALUES (?, ?, ?, ?, ?)",
                            tuple(ev.values()))
            self.db.commit()
        return ev

    def grace(self):
        """after the server (re)connects to the broker: give online nodes a fresh timeout"""
        with self.lock:
            self.db.execute("UPDATE nodes SET last_hb_epoch = ? WHERE state = 'online'", (time.time(),))
            self.db.commit()

    def check_timeouts(self, timeout_s):
        """online nodes whose last heartbeat is older than timeout_s become offline"""
        evs, now = [], time.time()
        with self.lock:
            for r in self.db.execute("SELECT node_id, motor_id, last_hb_epoch FROM nodes WHERE state = 'online'").fetchall():
                silent = now - (r["last_hb_epoch"] or 0)
                if silent > timeout_s:
                    ev = self._set_state(r["node_id"], r["motor_id"], "offline", f"no heartbeat for {silent:.0f} s",
                                         "went silent", f"no heartbeat for {silent:.0f} s (limit {timeout_s:g} s)")
                    evs.append(ev)
            self.db.commit()
        return [e for e in evs if e]

    def events(self, limit=50, node_id=None):
        q, args = "SELECT * FROM events", []
        if node_id:
            q, args = q + " WHERE node_id = ?", [node_id]
        with self.lock:
            rows = self.db.execute(q + " ORDER BY id DESC LIMIT ?", args + [limit]).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ reading
    _MOTOR_Q = ("SELECT m.*, n.state AS node_state, n.reason AS node_reason, n.state_since, n.last_heartbeat, "
                "n.last_hb_epoch, n.activity FROM motors m LEFT JOIN nodes n ON n.node_id = m.node_id")

    def motors(self):
        with self.lock:
            rows = self.db.execute(self._MOTOR_Q + " ORDER BY m.motor_id").fetchall()
        return [self._motor(r) for r in rows]

    def motor(self, motor_id):
        with self.lock:
            r = self.db.execute(self._MOTOR_Q + " WHERE m.motor_id = ?", (motor_id,)).fetchone()
        return self._motor(r) if r else None

    @staticmethod
    def _motor(r):
        run = json.loads(r["last_run"]) if r["last_run"] else None
        ver = json.loads(r["last_verdict"]) if r["last_verdict"] else None
        latest_run = run and {"seq": run["seq"], "at": run["created_at"], **run["payload"]}
        latest_verdict = ver and {"seq": ver["seq"], "at": ver["created_at"], **ver["payload"]}
        node_state = r["node_state"] or "no heartbeat yet"
        if node_state != "online":
            status = "UNKNOWN"
            detail = f"node {node_state}" + (f": {r['node_reason']}" if r["node_reason"] else "") + \
                     (f" (since {r['state_since']})" if r["state_since"] else "") + "; last known result shown"
        elif latest_verdict:
            v = latest_verdict["combined"]
            status = "HEALTHY" if v["verdict"] == "healthy" else "FAULTY"
            detail = f"{v['verdict']} ({v['confidence']}), verdict of cycle {latest_verdict['cycle']}"
        elif latest_run:
            status = "FAULTY" if latest_run["faulty"] else "CHECKING"
            detail = ("above the healthy level; fault type after the first full cycle" if latest_run["faulty"]
                      else "first cycle in progress")
        else:
            status, detail = "CHECKING", "no results yet"
        age = round(time.time() - r["last_hb_epoch"]) if r["last_hb_epoch"] else None
        return {"motor_id": r["motor_id"], "status": status, "status_detail": detail,
                "node_id": r["node_id"], "node_state": node_state, "last_heartbeat": r["last_heartbeat"],
                "heartbeat_age_s": age, "node_activity": json.loads(r["activity"]) if r["activity"] else None,
                "load": r["load"], "last_seen": r["last_seen"],
                "latest_run": latest_run, "latest_verdict": latest_verdict}

    def history(self, motor_id, type_=None, limit=50):
        q, args = "SELECT body FROM messages WHERE motor_id = ?", [motor_id]
        if type_:
            q += " AND type = ?"
            args.append(type_)
        with self.lock:
            rows = self.db.execute(q + " ORDER BY created_at DESC, seq DESC LIMIT ?", args + [limit]).fetchall()
        return [json.loads(r[0]) for r in rows]

    def nodes(self):
        with self.lock:
            rows = self.db.execute("SELECT * FROM nodes ORDER BY node_id").fetchall()
            out = []
            for r in rows:
                missing = self._missing(r)
                lost = json.loads(r["lost"] or "[]")
                out.append({"node_id": r["node_id"], "motor_id": r["motor_id"], "state": r["state"] or "no heartbeat yet",
                            "reason": r["reason"], "state_since": r["state_since"], "last_heartbeat": r["last_heartbeat"],
                            "activity": json.loads(r["activity"]) if r["activity"] else None,
                            "epoch": r["epoch"], "received": r["received"], "duplicates_ignored": r["duplicates"],
                            "highest_seq": r["max_seq"], "missing_seq": _ranges(missing), "missing_count": len(missing),
                            "lost_seq": _ranges(lost), "lost_count": len(lost), "run_id": r["run_id"],
                            "last_seen": r["last_seen"]})
        return out


def _ranges(nums):
    """[1,2,3,7,9,10] -> '1-3, 7, 9-10'"""
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else str(start))
    return ", ".join(out)
