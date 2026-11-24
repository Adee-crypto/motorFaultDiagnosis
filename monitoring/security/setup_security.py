"""
Security setup (step 8): makes the certificates and passwords the system uses.
Run it once from the repo root (it needs:  pip install cryptography):

    python monitoring/security/setup_security.py

It writes everything into monitoring/secrets/ (never pushed to git):

    ca.crt / ca.key        your own small certificate authority ("CA"). Everything
                           else is signed with ca.key; clients trust ca.crt. It can
                           only sign for localhost, broker, server and 127.0.0.1.
    broker.crt / .key      the broker's TLS certificate (names: broker, localhost)
    web.crt / .key         the dashboard's HTTPS certificate (names: localhost, server)
    mqtt/<user>.txt        one MQTT password per client: server, tools, healthcheck,
                           node-1 ... node-4 (a node's user name is its node id)
    mqtt_passwords.txt     all MQTT passwords together, for the broker only
    dashboard_users.txt    dashboard logins (hashed)
    dashboard_admin_password.txt   the first dashboard password (user "admin")

Later:
    python monitoring/security/setup_security.py add-node node-9     # MQTT login for a new node
    python monitoring/security/setup_security.py add-user alice      # dashboard login (asks for a password)
    python monitoring/security/setup_security.py remove-user alice
    python monitoring/security/setup_security.py list
    python monitoring/security/setup_security.py renew-certs         # new broker/web certificates, same CA

After add-node or renew-certs, restart the broker:  docker compose restart broker
"""
import argparse
import datetime as dt
import getpass
import ipaddress
import os
import secrets as rnd
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from security.userdb import NAME_OK, hash_password, load_users, save_users  # noqa: E402

SECRETS = os.path.join(os.path.dirname(HERE), "secrets")
MQTT_DIR = os.path.join(SECRETS, "mqtt")
FIXED_MQTT_USERS = ["server", "tools", "healthcheck"]
DEFAULT_NODES = ["node-1", "node-2", "node-3", "node-4"]

BROKER_NAMES = ["broker", "localhost", "127.0.0.1"]
CA_DNS_NAMES = ["localhost", "broker", "server"]       # the only names the CA can sign for
CA_IP_NETS = ["127.0.0.1/32"]
WEB_NAMES = ["localhost", "127.0.0.1", "server"]


def need_cryptography():
    try:
        import cryptography  # noqa: F401
    except ImportError:
        sys.exit("This needs the 'cryptography' package. Install it with:\n    pip install cryptography")


def p(*parts):
    return os.path.join(SECRETS, *parts)


def write_text(path, text, private=False):
    # newline="\n": the broker and containers are Linux, so no Windows line endings
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    if private and os.name != "nt":
        os.chmod(path, 0o600)


def write_bytes(path, data, private=False):
    with open(path, "wb") as fh:
        fh.write(data)
    if private and os.name != "nt":
        os.chmod(path, 0o600)


def read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


# ---------------------------------------------------------------- certificates
def _key():
    from cryptography.hazmat.primitives.asymmetric import ec
    return ec.generate_private_key(ec.SECP256R1())


def _pem_key(key):
    from cryptography.hazmat.primitives import serialization
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _pem_cert(cert):
    from cryptography.hazmat.primitives import serialization
    return cert.public_bytes(serialization.Encoding.PEM)


def make_ca():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.x509.oid import NameOID
    key = _key()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Motor monitoring local CA")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False,
                                         data_encipherment=False, key_agreement=False,
                                         encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            # Safety: this CA may only vouch for our own names. Even if ca.key were stolen,
            # certificates it signs for any other site (a bank, say) are refused by browsers.
            .add_extension(x509.NameConstraints(
                permitted_subtrees=[x509.DNSName(n) for n in CA_DNS_NAMES]
                + [x509.IPAddress(ipaddress.ip_network(n)) for n in CA_IP_NETS],
                excluded_subtrees=None), critical=True)
            .sign(key, hashes.SHA256()))
    write_bytes(p("ca.key"), _pem_key(key), private=True)
    write_bytes(p("ca.crt"), _pem_cert(cert))


def make_cert(stem, common_name, names, days=825):
    """A server certificate signed by our CA, valid for the given host names / IPs."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    with open(p("ca.key"), "rb") as fh:
        ca_key = serialization.load_pem_private_key(fh.read(), password=None)
    with open(p("ca.crt"), "rb") as fh:
        ca_cert = x509.load_pem_x509_certificate(fh.read())
    key = _key()
    alt = []
    for n in names:
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            alt.append(x509.DNSName(n))
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
            .issuer_name(ca_cert.subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    write_bytes(p(f"{stem}.key"), _pem_key(key), private=True)
    write_bytes(p(f"{stem}.crt"), _pem_cert(cert))


# ---------------------------------------------------------------- MQTT passwords
def mqtt_users():
    if not os.path.isdir(MQTT_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(MQTT_DIR) if f.endswith(".txt"))


def add_mqtt_user(user):
    path = os.path.join(MQTT_DIR, f"{user}.txt")
    if os.path.exists(path):
        return False
    write_text(path, rnd.token_urlsafe(24) + "\n", private=True)
    return True


def write_broker_password_list():
    """All users in one 'name:password' file. The broker hashes it when it starts."""
    lines = [f"{u}:{read_text(os.path.join(MQTT_DIR, u + '.txt'))}\n" for u in mqtt_users()]
    write_text(p("mqtt_passwords.txt"), "".join(lines), private=True)


# ---------------------------------------------------------------- commands
def cmd_init(a):
    need_cryptography()
    os.makedirs(MQTT_DIR, exist_ok=True)
    made = []
    if not os.path.exists(p("ca.crt")) or a.force:
        make_ca()
        made.append("certificate authority (ca.crt, ca.key)")
        a.force = True                  # a new CA means new certificates too
    for stem, cn, names in (("broker", "broker", BROKER_NAMES), ("web", "dashboard", WEB_NAMES)):
        if not os.path.exists(p(f"{stem}.crt")) or a.force:
            make_cert(stem, cn, names)
            made.append(f"{stem} certificate ({', '.join(names)})")
    for u in FIXED_MQTT_USERS + (a.nodes or DEFAULT_NODES):
        if add_mqtt_user(u):
            made.append(f"MQTT password for {u}")
    write_broker_password_list()

    users = load_users(p("dashboard_users.txt"))
    admin_pw = None
    if "admin" not in users:
        admin_pw = rnd.token_urlsafe(12)
        users["admin"] = hash_password(admin_pw)
        save_users(p("dashboard_users.txt"), users)
        write_text(p("dashboard_admin_password.txt"), admin_pw + "\n", private=True)
        made.append("dashboard user admin")

    print(f"Secrets folder: {SECRETS}")
    if made:
        print("Made:")
        for m in made:
            print(f"  - {m}")
    else:
        print("Everything already exists; nothing changed (use --force to start over).")
    if admin_pw:
        print(f"\nDashboard login:  user admin   password {admin_pw}")
        print("(also saved in monitoring/secrets/dashboard_admin_password.txt)")
    print("\nKeep this folder private. It is in .gitignore and is never pushed.")


def cmd_add_node(a):
    os.makedirs(MQTT_DIR, exist_ok=True)
    if not NAME_OK.match(a.name):
        sys.exit("Use letters, digits, '-', '_' or '.' only.")
    if add_mqtt_user(a.name):
        write_broker_password_list()
        print(f"MQTT login made for {a.name}: monitoring/secrets/mqtt/{a.name}.txt")
        print("Restart the broker so it knows the new login:  docker compose restart broker")
    else:
        print(f"{a.name} already has an MQTT login.")


def cmd_add_user(a):
    if not NAME_OK.match(a.name):
        sys.exit("Use letters, digits, '-', '_' or '.' only.")
    users = load_users(p("dashboard_users.txt"))
    if a.generate:
        pw = rnd.token_urlsafe(12)
    else:
        pw = getpass.getpass(f"New password for {a.name}: ")
        if len(pw) < 10:
            sys.exit("Use at least 10 characters.")
        if getpass.getpass("Type it again: ") != pw:
            sys.exit("The two passwords differ; nothing changed.")
    existed = a.name in users
    users[a.name] = hash_password(pw)
    os.makedirs(SECRETS, exist_ok=True)
    save_users(p("dashboard_users.txt"), users)
    print(f"{'Password changed' if existed else 'Dashboard user added'}: {a.name}")
    if a.generate:
        print(f"Password: {pw}")
    print("No restart needed: the server reads the users file at every login.")


def cmd_remove_user(a):
    users = load_users(p("dashboard_users.txt"))
    if a.name not in users:
        sys.exit(f"No dashboard user called {a.name}.")
    del users[a.name]
    save_users(p("dashboard_users.txt"), users)
    print(f"Removed {a.name}. Anyone signed in as {a.name} is signed out within a minute.")


def cmd_list(a):
    print("MQTT logins:     ", ", ".join(mqtt_users()) or "(none)")
    print("Dashboard users: ", ", ".join(sorted(load_users(p("dashboard_users.txt")))) or "(none)")
    if os.path.exists(p("ca.crt")):
        need_cryptography()
        from cryptography import x509
        for stem in ("ca", "broker", "web"):
            if os.path.exists(p(f"{stem}.crt")):
                with open(p(f"{stem}.crt"), "rb") as fh:
                    c = x509.load_pem_x509_certificate(fh.read())
                print(f"{stem + '.crt':16s}  valid until {c.not_valid_after_utc:%Y-%m-%d}")


def cmd_renew(a):
    need_cryptography()
    if not os.path.exists(p("ca.key")):
        sys.exit("No CA yet. Run setup first:  python monitoring/security/setup_security.py")
    make_cert("broker", "broker", BROKER_NAMES)
    make_cert("web", "dashboard", WEB_NAMES)
    print("New broker and web certificates made (same CA, so nothing else changes).")
    print("Restart:  docker compose restart broker server")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("init", help="make whatever is missing (the default)")
    s.add_argument("--nodes", nargs="+", help=f"node ids to make logins for (default: {' '.join(DEFAULT_NODES)})")
    s.add_argument("--force", action="store_true", help="make a new CA and new certificates (clients need the new ca.crt)")
    s = sub.add_parser("add-node"); s.add_argument("name")
    s = sub.add_parser("add-user"); s.add_argument("name")
    s.add_argument("--generate", action="store_true", help="make a random password and print it")
    s = sub.add_parser("remove-user"); s.add_argument("name")
    sub.add_parser("list")
    sub.add_parser("renew-certs")
    a = ap.parse_args()
    if a.cmd in (None, "init"):
        if a.cmd is None:
            a.nodes, a.force = None, False
        cmd_init(a)
    else:
        {"add-node": cmd_add_node, "add-user": cmd_add_user, "remove-user": cmd_remove_user,
         "list": cmd_list, "renew-certs": cmd_renew}[a.cmd](a)


if __name__ == "__main__":
    main()
