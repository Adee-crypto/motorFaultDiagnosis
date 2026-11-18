"""
central_store.py
=================
The central database (step 3): one SQLite file that holds every message
from every node, and each motor's latest status.

Tables
  messages   every message received, once. The key is msg_id
             ("<node_id>-<seq>"), so a message delivered twice by the
             broker (MQTT "at least once") is stored only once.
  motors     one row per motor: its node, load, when it was last heard
             from, its latest run result and its latest verdict.
  nodes      one row per node: messages received, highest message number,
             and which numbers are missing (lost or not yet sent).

All writes go through one lock, because the MQTT thread writes while the
web interface reads.
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CentralStore:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
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
        """)
        self.db.commit()

    # ------------------------------------------------------------ writing
    def add(self, m):
        """store one message; returns 'new' or 'duplicate'"""
        with self.lock:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (m["msg_id"], m["node_id"], m["motor_id"], m["seq"], m["type"], m["created_at"],
                 now_utc(), json.dumps(m)))
            new = cur.rowcount == 1
            self.db.execute("INSERT OR IGNORE INTO nodes (node_id) VALUES (?)", (m["node_id"],))
            if new:
                self.db.execute("UPDATE nodes SET received = received + 1, max_seq = MAX(max_seq, ?), "
                                "last_seen = ? WHERE node_id = ?", (m["seq"], now_utc(), m["node_id"]))
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
            if old is None or old["node_id"] != m["node_id"] or m["seq"] >= old["seq"]:
                self.db.execute(f"UPDATE motors SET {col} = ? WHERE motor_id = ?", (json.dumps(m), m["motor_id"]))

    # ------------------------------------------------------------ reading
    def motors(self):
        with self.lock:
            rows = self.db.execute("SELECT * FROM motors ORDER BY motor_id").fetchall()
        return [self._motor(r) for r in rows]

    def motor(self, motor_id):
        with self.lock:
            r = self.db.execute("SELECT * FROM motors WHERE motor_id = ?", (motor_id,)).fetchone()
        return self._motor(r) if r else None

    @staticmethod
    def _motor(r):
        run = json.loads(r["last_run"]) if r["last_run"] else None
        ver = json.loads(r["last_verdict"]) if r["last_verdict"] else None
        return {"motor_id": r["motor_id"], "node_id": r["node_id"], "load": r["load"], "last_seen": r["last_seen"],
                "latest_run": run and {"seq": run["seq"], "at": run["created_at"], **run["payload"]},
                "latest_verdict": ver and {"seq": ver["seq"], "at": ver["created_at"], **ver["payload"]}}

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
                have = {s for (s,) in self.db.execute("SELECT seq FROM messages WHERE node_id = ?", (r["node_id"],))}
                missing = [s for s in range(1, r["max_seq"] + 1) if s not in have]
                out.append({"node_id": r["node_id"], "received": r["received"], "duplicates_ignored": r["duplicates"],
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
