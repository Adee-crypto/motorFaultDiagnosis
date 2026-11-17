"""
messaging.py
=============
Sending node messages through an MQTT broker (step 2).

MQTT is a lightweight publish/subscribe protocol made for devices like
monitoring nodes. Nodes "publish" messages to a "topic"; anything that
"subscribes" to that topic receives them. The broker in the middle passes
them on, so nodes and the server never need to know each other's address.

Topics used:
    monitor/<node_id>/run_result     after every run
    monitor/<node_id>/verdict        after every full cycle
Subscribing to  monitor/#  receives everything.

Messages are sent with QoS 1 ("at least once"): the broker confirms each
one, and the node marks it as sent only after that confirmation. A message
may therefore arrive twice but is never silently lost while connected;
the msg_id lets the receiver ignore repeats (steps 3 and 5).
"""
import json
import threading

import paho.mqtt.client as mqtt

TOPIC_ROOT = "monitor"


def topic_for(message):
    return f"{TOPIC_ROOT}/{message['node_id']}/{message['type']}"


def parse_broker(text, default_port=1883):
    host, _, port = text.partition(":")
    return host or "localhost", int(port) if port else default_port


class Publisher:
    """One node's connection to the broker. Reconnects by itself in the background."""

    def __init__(self, broker, client_id, log=print, connect_wait=3.0):
        self.host, self.port = parse_broker(broker)
        self.log = log
        self.connected = threading.Event()
        self.closing = False
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.connect_async(self.host, self.port, keepalive=30)
        self.client.loop_start()
        if not self.connected.wait(connect_wait):
            self.log(f"broker {self.host}:{self.port} not reachable yet; results are kept locally, retrying in background")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if not reason_code.is_failure:
            self.connected.set()
            self.log(f"connected to broker {self.host}:{self.port}")

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

    def close(self):
        self.closing = True
        self.client.disconnect()
        self.client.loop_stop()
