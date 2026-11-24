"""
Dashboard logins (step 8): how passwords are stored and checked.

Passwords are never stored as they are. Each line of secrets/dashboard_users.txt
holds a name and a salted, slow hash of the password:

    admin:pbkdf2_sha256$260000$<salt>$<hash>

Checking a password repeats the same slow calculation and compares the result,
so a stolen users file doesn't reveal the passwords.
"""
import hashlib
import hmac
import os
import re

ITERATIONS = 260_000
NAME_OK = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


def hash_password(password, iterations=ITERATIONS):
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def check_password(password, stored):
    try:
        scheme, iterations, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                     bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


def load_users(path):
    """{name: stored hash}. A missing file means no users."""
    users = {}
    if not path or not os.path.exists(path):
        return users
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            name, stored = line.split(":", 1)
            users[name.strip()] = stored.strip()
    return users


def save_users(path, users):
    lines = ["# Dashboard logins (step 8). Made by security/setup_security.py; passwords are hashed.\n"]
    lines += [f"{name}:{stored}\n" for name, stored in sorted(users.items())]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(lines)
