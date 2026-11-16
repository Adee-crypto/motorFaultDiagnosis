"""
local_store.py
===============
The node's own small database: one SQLite file per node.

Every result the node produces is saved here as a "message": exactly what
will later be sent over the network (step 2). Saving first and sending
second is what will let a node keep working when the network is down
(step 5): the `sent` column is already here for that.

Table `messages`
    seq         1, 2, 3, ... per node, never reused, survives restarts
                (the server will use it to spot missing or repeated messages)
    msg_id      "<node_id>-<seq>", unique
    type        "run_result" (after every run) or "verdict" (after 8 runs)
    created_at  UTC time, ISO format
    body        the full message as JSON
    sent        0 = not yet sent to the server (used from step 5)

Table `meta` remembers which node and motor the file belongs to, so one
node's file can't be reused by another by mistake.
"""
import json
import os
import sqlite3


class LocalStore:
    def __init__(self, path, node_id, motor_id):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")          # safe if the node is killed mid-write
        self.db.execute("""CREATE TABLE IF NOT EXISTS messages (
                               seq INTEGER PRIMARY KEY, msg_id TEXT UNIQUE, type TEXT,
                               created_at TEXT, body TEXT, sent INTEGER DEFAULT 0)""")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._check_meta("node_id", node_id)
        self._check_meta("motor_id", motor_id)
        self.db.commit()

    def _check_meta(self, key, value):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO meta VALUES (?, ?)", (key, value))
        elif row[0] != value:
            raise ValueError(f"this store belongs to {key}={row[0]!r}, not {value!r}")

    def next_seq(self):
        row = self.db.execute("SELECT MAX(seq) FROM messages").fetchone()
        return (row[0] or 0) + 1

    def add(self, message):
        """save one message (a dict with seq, msg_id, type, created_at)"""
        self.db.execute("INSERT INTO messages (seq, msg_id, type, created_at, body) VALUES (?, ?, ?, ?, ?)",
                        (message["seq"], message["msg_id"], message["type"], message["created_at"],
                         json.dumps(message)))
        self.db.commit()

    def latest(self, n=20, type_=None):
        q, args = "SELECT body FROM messages", ()
        if type_:
            q, args = q + " WHERE type=?", (type_,)
        rows = self.db.execute(q + " ORDER BY seq DESC LIMIT ?", args + (n,)).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]

    def unsent(self):
        rows = self.db.execute("SELECT body FROM messages WHERE sent=0 ORDER BY seq").fetchall()
        return [json.loads(r[0]) for r in rows]

    def mark_sent(self, seq):
        self.db.execute("UPDATE messages SET sent=1 WHERE seq=?", (seq,))
        self.db.commit()

    def close(self):
        self.db.close()
