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
    msg_id      "<node_id>-<epoch>-<seq>", unique
    type        "run_result" (after every run) or "verdict" (after 8 runs)
    created_at  UTC time, ISO format
    body        the full message as JSON
    sent        0 = not yet sent to the server (used from step 5)

Table `meta` remembers which node and motor the file belongs to, so one
node's file can't be reused by another by mistake, and the store's "epoch":
a random id made when the file is created. Every message carries it, and
its msg_id is "<node_id>-<epoch>-<seq>". If the file is ever deleted or a
device is replaced, numbering restarts at 1 under a new epoch, so new
messages can never be mistaken for old ones (step 4).

Lock (step 5): a node holds a lock on "<file>.lock" while it runs, so a
second program can't use the same store at the same time (two programs
with one store would hand out the same message numbers twice). The
operating system releases the lock when the program ends, even after a
crash. Read-only tools (show_store.py) open the store without the lock.
"""
import json
import os
import sqlite3
import uuid

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class StoreInUse(RuntimeError):
    """another program is already using this store"""


def _lock(path):
    fh = open(path, "a+")
    try:
        if os.name == "nt":
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise StoreInUse(path)
    return fh


class LocalStore:
    def __init__(self, path, node_id, motor_id, lock=False):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lockfile = _lock(path + ".lock") if lock else None
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")          # safe if the node is killed mid-write
        self.db.execute("""CREATE TABLE IF NOT EXISTS messages (
                               seq INTEGER PRIMARY KEY, msg_id TEXT UNIQUE, type TEXT,
                               created_at TEXT, body TEXT, sent INTEGER DEFAULT 0)""")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._check_meta("node_id", node_id)
        self._check_meta("motor_id", motor_id)
        row = self.db.execute("SELECT value FROM meta WHERE key='epoch'").fetchone()
        if row is None:
            self.db.execute("INSERT INTO meta VALUES ('epoch', ?)", (uuid.uuid4().hex[:8],))
            row = self.db.execute("SELECT value FROM meta WHERE key='epoch'").fetchone()
        self.epoch = row[0]
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

    def get(self, seq):
        """one stored message by number, or None if this store doesn't have it"""
        row = self.db.execute("SELECT body FROM messages WHERE seq=?", (seq,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        self.db.close()
        if self._lockfile:
            self._lockfile.close()
            self._lockfile = None
