# Monitoring system (distributed / cloud layer)

This folder builds a monitoring system around the diagnosis code in the folder above, one step
at a time. The diagnosis code itself is not changed: the monitoring code only imports it.

## Step 1: simulated monitoring node (done)

A **node** is the small computer attached to one motor. Here it's a program that:

1. pretends to receive the motor's sensor data live, one second at a time, by replaying its CSV
   recordings;
2. after every run, works out the features and the Stage A check (is this run above the healthy
   level?) and saves a `run_result`;
3. after a full cycle (all 8 speed profiles), gives the three verdicts (triggers, ML, combined)
   and saves a `verdict`;
4. saves everything in its own small database (one SQLite file per node). These are the
   messages it will send over the network from step 2 on.

```
python monitoring/node.py --node-id node-1 --motor-id pump-A --data Loaded/RU --load 1
python monitoring/node.py --node-id node-2 --motor-id fan-B --data Unloaded/HH --load 0 --realtime --speed 5
python monitoring/show_store.py monitoring/stores/node-1.db
```

| Option | Meaning |
|---|---|
| `--node-id`, `--motor-id` | names of the node and the motor it watches |
| `--data` | folder with that motor's run CSVs (`X_X_<1-8>_<load>.csv`) |
| `--load` | 0 = unloaded, 1 = loaded |
| `--realtime`, `--speed` | replay at the recording's real pace (10 s per run), optionally sped up |
| `--cycles` | how many full cycles to run; 0 = keep going until Ctrl+C |
| `--store` | the node's database file (default `monitoring/stores/<node-id>.db`) |

### Files
| File | What it does |
|---|---|
| `node.py` | the simulated node: sensor replay, per-run Stage A, three verdicts per cycle |
| `local_store.py` | the node's local database. Every message gets a number (`seq`) that never repeats, even after a restart, and a `sent` flag for later steps |
| `show_store.py` | prints what a node has saved |
| `tests/check_step1.py` | checks that a node gives exactly the same results as `predict_all.py` for all 16 motors |

### Message format
Every result is a JSON message, already in the form it will be sent in:
```json
{"msg_id": "node-1-9", "seq": 9, "type": "verdict", "node_id": "node-1", "motor_id": "pump-A",
 "load": 1, "created_at": "2026-11-16T20:25:13+00:00",
 "payload": {"cycle": 1, "runs": 8, "faulty_runs": 8,
             "triggers": {"verdict": "rotor_unbalance", "confidence": "clear", "points": {...}},
             "ml": {...}, "combined": {...}, "agree": true}}
```
A node sends results only, never raw signals: about 1 KB per run instead of about 20 MB.

## Coming next
2. Messaging through an MQTT broker  3. Central server and database  4. Failure detection
5. Store-and-forward  6. Dashboard  7. One-command startup  8. Security  9. Cloud
