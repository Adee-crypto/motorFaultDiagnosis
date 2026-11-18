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
                          ("epoch", "TEXT DEFAULT ''"), ("epoch_started", "TEXT")):
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
                node = self.db.execute("SELECT epoch, epoch_started FROM nodes WHERE node_id = ?", (m["node_id"],)).fetchone()
                if (node["epoch"] or "") != epoch and (node["epoch_started"] or "") <= m["created_at"]:
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

    def heartbeat(self, hb):
        """a heartbeat arrived: the node is online; returns an event if that is news"""
        with self.lock:
            self._ensure_node(hb["node_id"], hb.get("motor_id"))
            info = {k: hb.get(k) for k in ("state", "cycle", "runs_done", "runs_per_cycle", "unsent", "uptime_s", "interval_s")}
            self.db.execute("UPDATE nodes SET last_heartbeat = ?, last_hb_epoch = ?, activity = ? WHERE node_id = ?",
                            (now_utc(), time.time(), json.dumps(info), hb["node_id"]))
            ev = self._set_state(hb["node_id"], hb.get("motor_id"), "online", "heartbeats arriving",
                                 "online", "heartbeats arriving")
            self.db.commit()
        return ev

    def node_status(self, st):
        """a status message: "online" when the node connects, "offline" on goodbye or last will"""
        with self.lock:
            self._ensure_node(st["node_id"], st.get("motor_id"))
            if st["status"] == "online":
                self.db.execute("UPDATE nodes SET last_hb_epoch = ? WHERE node_id = ?", (time.time(), st["node_id"]))
                ev = self._set_state(st["node_id"], st.get("motor_id"), "online", "connected", "online", "connected")
            elif st.get("reason") == "connection lost":
                ev = self._set_state(st["node_id"], st.get("motor_id"), "offline", "connection lost",
                                     "connection lost", "broker announced the node's last will (no goodbye)")
            else:
                ev = self._set_state(st["node_id"], st.get("motor_id"), "stopped", st.get("reason"),
                                     "stopped", f"node said goodbye: {st.get('reason')}")
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
                have = {s for (s,) in self.db.execute("SELECT seq FROM messages WHERE node_id = ? AND epoch = ?",
                                                      (r["node_id"], r["epoch"] or ""))}
                missing = [s for s in range(1, r["max_seq"] + 1) if s not in have]
                out.append({"node_id": r["node_id"], "motor_id": r["motor_id"], "state": r["state"] or "no heartbeat yet",
                            "reason": r["reason"], "state_since": r["state_since"], "last_heartbeat": r["last_heartbeat"],
                            "activity": json.loads(r["activity"]) if r["activity"] else None,
                            "epoch": r["epoch"], "received": r["received"], "duplicates_ignored": r["duplicates"],
                            "highest_seq": r["max_seq"], "missing_seq": _ranges(missing), "missing_count": len(missing),
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
