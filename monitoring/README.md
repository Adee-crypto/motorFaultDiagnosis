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
{"msg_id": "node-1-3f9a12c0-9", "seq": 9, "epoch": "3f9a12c0", "type": "verdict", "node_id": "node-1", "motor_id": "pump-A",
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
- **Messages lost at the broker:** recovered from step 5 on, by asking the node to resend them.
- `listener.py` from step 2 still works next to the server, as a live view of the messages.

| File | What it does |
|---|---|
| `server/server.py` | the server: MQTT subscriber + web API |
| `server/central_store.py` | the central database (SQLite file `server/central.db`) |
| `status.py` | asks the server for every motor's status and prints a table |

## Step 4: failure detection (done)

The server now knows whether each node is alive, so a crashed, frozen or disconnected node is
never mistaken for a healthy motor.

```
python monitoring/server/server.py                        # node offline after 15 s without a heartbeat
python monitoring/node.py --node-id node-1 --motor-id pump-A --data Loaded/RU --load 1 --broker localhost --cycles 0 --realtime --speed 5
python monitoring/status.py --events --watch 2            # live table of statuses and recent events
```

How it works
- **Heartbeats:** every 5 s (`--heartbeat`) each node sends a small "I'm alive" message with what
  it is doing (recording, analysing, diagnosing, idle), its progress through the cycle and how
  many results are still unsent.
- **Last will:** when a node connects, it leaves an "offline, connection lost" message with the
  broker. If the connection breaks without a goodbye (crash, network loss), the broker
  publishes it for the node: a crash is noticed within seconds. A node that stops normally
  says goodbye itself ("finished" or "stopped by user").
- **Watchdog:** a node counts as online only while its heartbeats keep arriving. No heartbeat
  for 15 s (`--timeout` on the server) makes it offline. This catches a frozen node that is still
  connected, which the last will can't. The server uses its own clock, since node clocks may
  differ, and pauses the watchdog while it can't reach the broker itself.
- **Motor status:** HEALTHY or FAULTY (node online, from its latest verdict), CHECKING (node
  online, no verdict yet) or **UNKNOWN** (node offline or stopped). For an UNKNOWN motor the last
  known verdict is shown with its time, but never as the current status.
- **Events:** every change of a node's state (online, went silent, connection lost, stopped,
  numbering restarted) is recorded with its time and reason: `GET /events`, `status.py --events`.
- **Message numbering restarts:** each node store has a random "epoch" id, part of every message
  id (`node-1-<epoch>-<seq>`). If a node's store is deleted or the device is replaced, its new
  messages can't be mistaken for old ones; the server records a "numbering restarted" event.

Testing failures: the node can fake one.
```
python monitoring/node.py ... --broker localhost --cycles 0 --realtime --speed 5 --fail-after 20 --fail-mode crash
python monitoring/node.py ... --broker localhost --cycles 0 --realtime --speed 5 --fail-after 20 --fail-mode hang
```
`crash` kills the program at once (no goodbye): detected through the last will, within seconds.
`hang` freezes it but keeps the connection open: detected only by the missing heartbeats,
after the timeout. Stop a frozen node with Ctrl+C.

Changes to existing files: `messaging.py` (heartbeat, last will, online/offline), `node.py`
(heartbeat thread, test failures, epoch), `local_store.py` (epoch), `server/central_store.py`
(node states, events, epochs; an existing step 3 database is upgraded automatically),
`server/server.py` (watchdog, `/events`, `--timeout`), `status.py` (status column, `--events`,
`--watch`), `listener.py` (shows online/offline; `--heartbeats` to show heartbeats).

## Step 5: store-and-forward (done)

No result is lost for good without the system knowing about it.

- **Backlog:** results a node couldn't send (broker down, network cut, a crash just after saving)
  stay in its store as "not sent". As soon as the connection is back, and when the node starts,
  it sends them, oldest first, before new results. At the end of a run it keeps trying for
  10 s (`--flush-wait`); anything still unsent waits for the next run.
- **Resend requests:** the server knows each node's message numbers. Every 20 s, for every online
  node with gaps, it asks the node (`control/<node_id>/resend`) to send those numbers again. This
  recovers messages that reached the broker but never reached the server, for example sent while
  the server was down and not kept by the broker. The first request comes ~10 s after a node comes
  online, so its own backlog arrives first. A resent message that had arrived after all is ignored
  as a duplicate.
- **Lost for good:** if the node no longer has a number (its store was wiped or replaced), it says
  so. The server records those numbers once, with a "lost for good" event, and stops asking.
- **One program per node:** a node locks its store while it runs. A second program with the same
  `--node-id` on the same computer refuses to start: "node-2 is already running on this computer".
- **Duplicate node names:** every run of a node program has its own id. If two programs report under
  one node name (e.g. on two computers), the server records one "duplicate node id" event instead of
  the connection lost / online flip-flop seen in step 4.

```
python monitoring/status.py --nodes --events     # per node: unsent, missing (asked for), lost for good
```

| Situation | What happens |
|---|---|
| Broker down while the node works | results kept, then sent in order when the broker is back |
| Server down (broker doesn't keep its messages) | server sees the gap with the next message, asks, node resends |
| Node crashed after saving, before sending | sent when the node starts again |
| Part of a node's store wiped | the rest is resent; the missing part is recorded as lost for good |
| Second program with the same node id | same computer: refused; elsewhere: one "duplicate node id" event |

Changes to existing files: `local_store.py` (lock, fetch by number), `messaging.py` (run id,
reconnect signal, resend requests on `control/<node_id>/resend`), `node.py` (backlog, answering
requests, `--flush-wait`), `server/central_store.py` and `server/server.py` (requests, lost for good,
duplicate names), `status.py` and `listener.py` (show the new information).

## Step 6: dashboard (done)

Open **http://localhost:8000** while the server runs. The API is still at
http://localhost:8000/docs.

The dashboard is one web page (`server/static/dashboard.html`) that the server
hands out. It needs no internet. Every 2 seconds it reads `/health`,
`/motors`, `/nodes` and `/events` and redraws:

- **Annunciator (top row):** one window per motor. It is grey when the motor is
  healthy, outlined while the first cycle is running (CHECKING), red when the
  motor is FAULTY and amber when its status is UNKNOWN (node offline). A
  window flashes until its alarm is acknowledged. Click a window to see that
  motor's details.
- **Alarms:** faulty motors, unknown status, the two methods disagreeing, results
  lost for good, and a duplicate node id. "Acknowledge" stops the flashing; the
  browser remembers it. A new problem (a different fault, a new outage) flashes
  again.
- **Motors:** each node's state, heartbeat age (shown in red when it is late), what
  it is doing, its progress through the cycle (8 bars) and its last verdict.
- **Selected motor:** the triggers, ML and combined verdicts; points per fault;
  a chart of vibration (accel1) and sound (mic) against the healthy level for
  the last 16 runs, with the fault level at 2×; recent runs; earlier verdicts.
- **Nodes:** received, highest message number, unsent, missing (being asked
  for again), lost for good, and repeats ignored.
- **Events:** every node coming online, going offline or silent, stopping, etc.

If the server stops, a banner says so and the page keeps the last data. If the
server loses the broker, a banner says the statuses are frozen. A Light/Dark
button switches the theme.

## Step 7: one-command startup with Docker (done)

Everything (broker, server + dashboard, and four simulated nodes) starts with
one command. Run it from the `monitoring` folder:

```
docker compose up -d --build
```

Then open http://localhost:8000 (https from step 8 on). The first time takes a few minutes, because
Docker downloads Python and the libraries and builds the image.

| Command (in `monitoring`) | What it does |
|---|---|
| `docker compose up -d --build` | start everything (rebuilds the image if the code changed) |
| `docker compose ps` | what is running |
| `docker compose logs -f` / `docker compose logs -f node-1` | follow the output of all / one |
| `docker compose stop node-2` / `docker compose start node-2` | stop / start one part |
| `docker compose restart server` | restart the server (its database is kept) |
| `docker compose down` | stop and remove everything; the data is kept |
| `docker compose down -v` | also wipe the data (database, node stores, broker queue) |
| `docker compose up -d broker` | only the broker, as in steps 2 to 6 |

How it is put together:

- `Dockerfile`: one image (`motor-monitor`) for the server and the nodes:
  Python, the libraries (`requirements.txt`, `requirements-diagnosis.txt`), the
  diagnosis code and `monitoring/`. `Dockerfile.dockerignore` keeps everything
  else out. The recordings are not in the image: the nodes see `Loaded/` and
  `Unloaded/` read-only through a mount.
- `docker-compose.yml`: the broker starts first; the server and nodes wait until
  it is healthy (accepts connections), and the nodes also start after the
  server (and stop before it). Inside Docker they reach the broker by its
  service name, `broker`.
- Data lives in Docker volumes, so it survives restarts and `docker compose down`:
  `server-data` (central database), `node-stores` (each node's store from
  step 5), `broker-data` (the broker's queue).
- `restart: unless-stopped`: if a program crashes, Docker starts it again. A
  node that comes back carries on numbering from its store and sends what it
  had kept. (A container you stop yourself stays stopped.)
- `docker stop` / `docker compose down` let each node finish cleanly: it sends
  what it can and says "stopped", so the dashboard doesn't report a lost connection.

To add a motor, copy a node block in `docker-compose.yml`, give it a new
service name, `--node-id`, `--motor-id` and `--data` folder, then run
`docker compose up -d`.

Test failures inside a container with `tests/crash_node.py`:

```
docker compose exec node-3 python monitoring/tests/crash_node.py            # crash: Docker restarts it
docker compose exec node-3 python monitoring/tests/crash_node.py --hang 30  # freeze for 30 s
```

The dashboard measures heartbeat ages on the server's clock, so a Docker
clock that is a few seconds off from Windows doesn't make nodes look late.

## Step 8: security (done)

Three locks:

1. **Encryption (TLS).** All MQTT traffic goes over TLS on port 8883 (the
   unencrypted port 1883 is gone), and the dashboard uses HTTPS:
   **https://localhost:8000**. Every client checks the other side's certificate
   against our own certificate authority (`secrets/ca.crt`), so nobody can listen
   in or pretend to be the broker or the server. The CA can only sign for
   `localhost`, `broker`, `server` and `127.0.0.1`, so even a stolen `ca.key` can't
   fake any other website.
2. **Logins.** The broker refuses clients without a valid user name and
   password. Every client has its own: `server`, `tools` (read-only, for
   `listener.py`), `healthcheck`, and one per node, whose user name is its node id.
   The dashboard and API need a sign-in.
3. **Topic rules** (`broker/acl`). Each login may only use its own topics. A node may
   only write `monitor/<its id>/...` and read `control/<its id>/...`, so a stolen
   node password can't fake another motor's results or read other nodes'
   messages. `tools` may only read.

### First-time setup

From the repo root:

```
pip install cryptography
python monitoring/security/setup_security.py
```

This makes `monitoring/secrets/` (in `.gitignore`, never pushed, never put in the
image) and prints the first dashboard password (user `admin`). It is also in
`secrets/dashboard_admin_password.txt`.

To make your browser trust the dashboard, add the CA to Windows once:

```
Import-Certificate -FilePath monitoring\secrets\ca.crt -CertStoreLocation Cert:\CurrentUser\Root
```

Windows asks you to confirm. Chrome and Edge then show the padlock. If you skip
this, the browser warns that the page is "not private"; you can continue anyway.
To remove it later:

```
Get-ChildItem Cert:\CurrentUser\Root | Where-Object Subject -like "*Motor monitoring local CA*" | Remove-Item
```

### Running

`docker compose up -d --build` as before. Each container gets only the secrets it
needs (Docker `secrets:`, mounted read-only under `/run/secrets/`): a node gets
just `ca.crt` and its own password. When the broker starts, it turns the
password list into hashes with Mosquitto's own tool.

The tools on your PC find the secrets by themselves:

```
python monitoring/listener.py            # TLS, user tools (read-only)
python monitoring/status.py              # HTTPS, user admin
python monitoring/node.py --node-id node-9 ... --broker localhost   # needs a login for node-9 first
```

`--no-tls` connects the old way, to an unsecured broker as in steps 2 to 7.

### Managing logins

| Command (from the repo root) | What it does |
|---|---|
| `python monitoring/security/setup_security.py add-user alice` | new dashboard user (asks for a password); no restart needed |
| `python monitoring/security/setup_security.py remove-user alice` | removes them; signed out within a minute |
| `python monitoring/security/setup_security.py add-node node-5` | MQTT login for a new node; then `docker compose restart broker` |
| `python monitoring/security/setup_security.py list` | logins and certificate expiry dates |
| `python monitoring/security/setup_security.py renew-certs` | new broker/web certificates (same CA); then restart broker and server |

### What the dashboard login does

- Passwords are stored hashed (PBKDF2, 260,000 rounds), never as they are.
- After signing in, the browser gets a random session cookie: HttpOnly (page
  scripts can't read it), Secure (HTTPS only) and SameSite=Strict (never sent
  by other websites). It lasts 12 hours. "Sign out" ends it on the server too.
- 5 wrong passwords from one address lock sign-in from it for 60 seconds.
- Scripts (status.py) send the name and password with each request, over HTTPS.
- Pages are sent with headers that stop other sites from framing them or
  loading scripts into them.

### Tests run

- Refused at the broker: unencrypted connections, no password, wrong password,
  and a client trusting a different CA.
- Blocked by the topic rules: node-1 posting as node-2, node-1 writing control
  messages, node-1 reading another node's control topic, `tools` writing.
- Still working: results, heartbeats, last wills, and resend requests (server
  to node).
- Dashboard: plain http doesn't answer. Without signing in, pages go to the
  sign-in page and the API returns 401. A wrong password says so; the 5th
  locks sign-in for 60 s. A cookie stops working after "Sign out". A removed
  user is signed out within a minute. A "next" link to another site is ignored.

## Coming next
9. Cloud
