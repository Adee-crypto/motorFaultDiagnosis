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

## Step 2: messaging through an MQTT broker (done)


Nodes now send each message through a **broker**, a program that passes messages from senders
to whoever is listening. Nodes and the server never need to know each other's address; they
only know the broker. The broker is **Mosquitto**, run in Docker.

```
pip install -r monitoring/requirements.txt                    # once: the MQTT library
docker compose -f monitoring/docker-compose.yml up -d         # start the broker
python monitoring/listener.py                                 # terminal 1: watch messages arrive
python monitoring/node.py --node-id node-1 --motor-id pump-A --data Loaded/RU --load 1 --broker localhost
python monitoring/node.py --node-id node-2 --motor-id fan-B --data Unloaded/HH --load 0 --broker localhost
docker compose -f monitoring/docker-compose.yml down          # stop the broker
```

How it works
- **Topics:** a node publishes to `monitor/<node_id>/run_result` and `monitor/<node_id>/verdict`;
  the listener subscribes to `monitor/#`, which means everything.
- **Saved first, then sent:** a message is saved in the node's local store before sending, and
  marked "sent" only after the broker confirms it (MQTT QoS 1, "at least once").
- **Broker down:** the node keeps working, keeps the results locally as "not sent", and
  reconnects by itself when the broker is back. Sending the backlog is step 5.
- **Listener:** prints every message and flags gaps, repeats and out-of-order numbers in a
  node's message numbers, the first sign of a lost or doubled message.

| File | What it does |
|---|---|
| `messaging.py` | the node's connection to the broker: connect, reconnect, send with confirmation |
| `listener.py` | shows every message arriving at the broker; stands in for the server until step 3 |
| `docker-compose.yml` | starts the broker (later steps add the server, database and dashboard here) |
| `broker/mosquitto.conf` | broker settings: port 1883, no password yet (step 8 adds security) |
| `requirements.txt` | Python packages the monitoring code needs |

## Step 3: central server and database (done)

The **server** receives every message from every node through the broker, saves it in one
central database, and answers questions about the motors over HTTP (a web API).

```
pip install -r monitoring/requirements.txt                    # once: adds fastapi and uvicorn
docker compose -f monitoring/docker-compose.yml up -d         # broker (as in step 2)
python monitoring/server/server.py                            # terminal 1: the server
python monitoring/node.py --node-id node-1 --motor-id pump-A --data Loaded/RU --load 1 --broker localhost
python monitoring/status.py --nodes                           # every motor's status, as a table
```
Or open **http://localhost:8000/docs** in a browser and try the questions there.

| Question (GET) | Answer |
|---|---|
| `/motors` | every motor: its node, load, when last heard from, latest run, latest verdict |
| `/motors/{motor_id}` | one motor |
| `/motors/{motor_id}/history?type=verdict&limit=20` | its recent messages |
| `/nodes` | per node: messages received, duplicates ignored, highest message number, missing numbers |
| `/health` | is the server up and connected to the broker |

How it works
- **Each message stored once:** the database key is the message id (`node-1-9`), so a message
  delivered twice (MQTT "at least once") is stored once and counted as a duplicate.
- **Latest status per motor:** updated with each new message. A late, older message from the same
  node doesn't overwrite a newer one.
- **Missing messages:** per node, the server knows the highest message number it has seen and
  which numbers below it never arrived.
- **Server offline:** the server keeps a lasting session with the broker (client id
  `monitor-server`, clean session off), so Mosquitto keeps its messages while it's away and hands
  them over when it reconnects.
- **Known gap, fixed in step 5:** if the broker itself loses queued messages, the node has already
  marked them sent. The server sees the missing numbers but can't get them back yet.
- `listener.py` from step 2 still works next to the server, as a live view of the messages.

| File | What it does |
|---|---|
| `server/server.py` | the server: MQTT subscriber + web API |
| `server/central_store.py` | the central database (SQLite file `server/central.db`) |
| `status.py` | asks the server for every motor's status and prints a table |

## Coming next
4. Failure detection  5. Store-and-forward  6. Dashboard  7. One-command startup  8. Security
9. Cloud
