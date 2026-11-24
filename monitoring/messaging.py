"""
messaging.py
=============
A node's connection to the MQTT broker (steps 2 and 4).

MQTT is a lightweight publish/subscribe protocol made for devices like
monitoring nodes. Nodes "publish" messages to a "topic"; anything that
"subscribes" to that topic receives them. The broker in the middle passes
them on, so nodes and the server never need to know each other's address.

Topics used:
    monitor/<node_id>/run_result     after every run                (QoS 1)
    monitor/<node_id>/verdict        after every full cycle         (QoS 1)
    monitor/<node_id>/heartbeat      "I'm alive", every few seconds (QoS 0)
    monitor/<node_id>/status         "online" / "offline", kept by the broker (retained)
    monitor/<node_id>/resend_reply   answer to a resend request: numbers it no longer has (step 5)
    control/<node_id>/resend         server -> node: "please send these message numbers again" (step 5)
Subscribing to  monitor/#  receives everything the nodes send.

Results are sent with QoS 1 ("at least once"): the broker confirms each
one, and the node marks it as sent only after that confirmation.

Failure detection (step 4):
  - heartbeat: a small message every few seconds. If the server stops
    receiving them, the node is treated as offline, even if it is still
    connected (for example a frozen program).
  - last will: when the node connects it leaves an "offline, connection
    lost" message with the broker. If the connection breaks without a
    proper goodbye (crash, network loss), the broker publishes it for the
    node. On a normal stop the node says "offline, stopped" itself.
  - status messages are "retained": the broker keeps the latest one per
    node and gives it to anyone who subscribes later, e.g. a restarted server.

Store-and-forward (step 5):
  - run_id: a random id for every run of the node program, sent with its
    heartbeats and status, so the server can tell a restart from two
    programs using the same node name at once.
  - `reconnected` is set each time the connection comes (back) up, so the
    node knows to send its backlog of unsent results.
  - resend requests from the server arrive on control/<node_id>/resend and
    are handed to the node through `requests` (a queue), because the node
    reads its store only from its main thread.

Security (step 8):
  - TLS: the connection to the broker is encrypted (port 8883), and the
    client checks the broker's certificate against our own certificate
    authority (secrets/ca.crt), so it can't be fooled by a fake broker.
  - login: every client has a user name and password (secrets/mqtt/<user>.txt).
    A node's user name is its node id; the broker's topic rules (broker/acl)
    let each node write only its own topics.
  - Without secrets (or with --no-tls) a client connects the old way, without
    encryption or login, to a broker set up as in steps 2 to 7.
"""
import datetime as dt
import json
import os
import queue
import socket
import ssl
import threading
import uuid

import paho.mqtt.client as mqtt

TOPIC_ROOT = "monitor"
CONTROL_ROOT = "control"
KEEPALIVE = 10   # seconds; the broker notices a silently dead connection after ~1.5x this
SECRETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets")
TLS_PORT, PLAIN_PORT = 8883, 1883


class Security:
    """How a client proves who it is and protects the connection (step 8)."""

    def __init__(self, ca=None, user=None, password=None):
        self.ca, self.user, self.password = ca, user, password

    @property
    def tls(self):
        return bool(self.ca)

    @property
    def default_port(self):
        return TLS_PORT if self.tls else PLAIN_PORT

    def apply(self, client):
        if self.ca:
            # checks the broker's certificate and that it was issued for the host name we dial
            client.tls_set(ca_certs=self.ca, cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
        if self.user:
            client.username_pw_set(self.user, self.password)

    def describe(self):
        if not self.tls:
            return "NOT encrypted, no login" + (f" (user {self.user})" if self.user else "")
        return f"TLS, user {self.user}"

    def diagnose(self, host, port):
        """Why can't we connect? Tries the connection by hand to get a readable reason."""
        try:
            with socket.create_connection((host, port), timeout=3) as raw:
                if not self.tls:
                    return "the broker accepts the connection; check the login"
                ctx = ssl.create_default_context(cafile=self.ca)
                with ctx.wrap_socket(raw, server_hostname=host):
                    return "TLS is fine; the broker refused the login (check the user name and password)"
        except ssl.SSLCertVerificationError as e:
            return (f"the broker's certificate doesn't match ({e.verify_message}). Is ca.crt from the same "
                    f"setup as the broker's certificate, and is the host name one of: broker, localhost, 127.0.0.1?")
        except ssl.SSLError as e:
            return f"TLS failed ({e.reason or e}). Is this the TLS port (8883)?"
        except ConnectionRefusedError:
            return f"nothing is listening on {host}:{port}. Is the broker running? (docker compose ps)"
        except OSError as e:
            return f"can't reach {host}:{port} ({e})"


def add_security_args(ap):
    g = ap.add_argument_group("security (step 8)")
    g.add_argument("--ca", help="certificate authority to trust (default: monitoring/secrets/ca.crt)")
    g.add_argument("--user", help="MQTT user name (see each program's default)")
    g.add_argument("--password-file", help="file holding the MQTT password (default: monitoring/secrets/mqtt/<user>.txt)")
    g.add_argument("--no-tls", action="store_true",
                   help="connect without encryption or login (only for an unsecured broker as in steps 2 to 7)")
    return g


def security_from_args(a, default_user):
    """Security settings from the command line, with defaults from monitoring/secrets/."""
    if getattr(a, "no_tls", False):
        return Security(user=a.user) if a.user else Security()
    ca = a.ca or os.path.join(SECRETS, "ca.crt")
    if not os.path.exists(ca):
        raise SystemExit(f"No certificate authority found ({ca}).\n"
                         f"Make the secrets first:  python monitoring/security/setup_security.py\n"
                         f"(or add --no-tls to talk to an unsecured broker as in steps 2 to 7)")
    user = a.user or default_user
    pw_file = a.password_file or os.path.join(SECRETS, "mqtt", f"{user}.txt")
    if not os.path.exists(pw_file):
        raise SystemExit(f"No MQTT password for user {user} ({pw_file}).\n"
                         f"Make one with:  python monitoring/security/setup_security.py add-node {user}\n"
                         f"then restart the broker:  docker compose restart broker")
    with open(pw_file, encoding="utf-8") as fh:
        password = fh.read().strip()
    return Security(ca, user, password)


def topic_for(message):
    return f"{TOPIC_ROOT}/{message['node_id']}/{message['type']}"


def parse_broker(text, default_port=PLAIN_PORT):
    host, _, port = text.partition(":")
    return host or "localhost", int(port) if port else default_port


def now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Publisher:
    """One node's connection to the broker. Reconnects by itself in the background."""

    def __init__(self, broker, client_id, log=print, connect_wait=3.0, motor_id=None, security=None):
        self.security = security or Security()
        self.host, self.port = parse_broker(broker, self.security.default_port)
        self._last_problem = None
        self.node_id, self.motor_id = client_id, motor_id
        self.log = log
        self.connected = threading.Event()
        self.reconnected = threading.Event()
        self.requests = queue.Queue()
        self.run_id = uuid.uuid4().hex[:6]
        self.closing = False
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.on_connect_fail = self._on_connect_fail
        self.security.apply(self.client)                    # TLS + login (step 8)
        # last will: published by the broker if this connection dies without a goodbye
        self.client.will_set(self._status_topic(), self._status("offline", "connection lost"), qos=1, retain=True)
        self.client.connect_async(self.host, self.port, keepalive=KEEPALIVE)
        self.client.loop_start()
        if not self.connected.wait(connect_wait):
            self.log(f"broker {self.host}:{self.port} not reachable yet; results are kept locally, retrying in background")

    def _status_topic(self):
        return f"{TOPIC_ROOT}/{self.node_id}/status"

    def _status(self, status, reason):
        return json.dumps({"type": "status", "node_id": self.node_id, "motor_id": self.motor_id,
                           "run_id": self.run_id, "status": status, "reason": reason, "at": now_utc()})

    def _problem(self, text):
        if text != self._last_problem:                      # say each new problem once, not every retry
            self._last_problem = text
            self.log(text)

    def _on_connect_fail(self, client, userdata):
        self._problem(f"can't connect to the broker at {self.host}:{self.port}: "
                      f"{self.security.diagnose(self.host, self.port)}; retrying")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            self._problem(f"the broker refused the login as {self.security.user!r}: {reason_code}; retrying")
        else:
            self._last_problem = None
            self.connected.set()
            self.reconnected.set()
            client.subscribe(f"{CONTROL_ROOT}/{self.node_id}/resend", qos=1)
            client.publish(self._status_topic(), self._status("online", "connected"), qos=1, retain=True)
            self.log(f"connected to broker {self.host}:{self.port} ({self.security.describe()})")

    def _on_message(self, client, userdata, msg):
        """only real resend requests for this node are passed on; anything else is ignored"""
        if msg.topic != f"{CONTROL_ROOT}/{self.node_id}/resend":
            return
        try:
            body = json.loads(msg.payload)
        except ValueError:
            return
        if isinstance(body, dict) and body.get("type") == "resend_request" and isinstance(body.get("seqs"), list):
            self.requests.put(body)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        if self.connected.is_set() and not self.closing:
            self.log(f"lost connection to broker ({reason_code}); results are kept locally")
        self.connected.clear()

    def send(self, message, timeout=5.0):
        """True once the broker has confirmed the message; False if not connected or no confirmation."""
        if not self.connected.is_set():
            return False
        try:
            info = self.client.publish(topic_for(message), json.dumps(message), qos=1)
            info.wait_for_publish(timeout)
            return info.is_published()
        except (RuntimeError, ValueError):
            return False

    def heartbeat(self, info):
        """fire-and-forget "I'm alive" (QoS 0): a lost heartbeat is simply replaced by the next one"""
        if self.connected.is_set():
            body = {"type": "heartbeat", "node_id": self.node_id, "motor_id": self.motor_id, "run_id": self.run_id,
                    "at": now_utc(), **info}
            self.client.publish(f"{TOPIC_ROOT}/{self.node_id}/heartbeat", json.dumps(body), qos=0)

    def reply(self, body):
        """answer a resend request (e.g. which numbers this node no longer has)"""
        if self.connected.is_set():
            body = {"type": "resend_reply", "node_id": self.node_id, "motor_id": self.motor_id,
                    "run_id": self.run_id, "at": now_utc(), **body}
            self.client.publish(f"{TOPIC_ROOT}/{self.node_id}/resend_reply", json.dumps(body), qos=1)

    def close(self, reason="stopped"):
        """normal stop: say goodbye, so the broker does not send the last will"""
        self.closing = True
        if self.connected.is_set():
            try:
                self.client.publish(self._status_topic(), self._status("offline", reason),
                                    qos=1, retain=True).wait_for_publish(3)
            except (RuntimeError, ValueError):
                pass
        self.client.disconnect()
        self.client.loop_stop()
