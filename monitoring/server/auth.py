"""
Dashboard and API login (step 8).

- People sign in on /login with a name and password from
  secrets/dashboard_users.txt (passwords stored hashed, see security/userdb.py).
  They get a session cookie: random, HttpOnly (page scripts can't read it),
  Secure (only sent over HTTPS) and SameSite=Strict (never sent by other
  websites). A session lasts 12 hours.
- Scripts such as status.py send the name and password with every request
  instead (HTTP "Basic" authentication). That is only safe over HTTPS, which
  the server uses whenever the certificates exist.
- After 5 wrong passwords from one address, logins from it are refused for
  60 seconds, so passwords can't be guessed quickly.
- The users file is read again at every login and every minute for open
  sessions, so adding or removing a user needs no restart.
"""
import base64
import html
import os
import secrets
import sys
import threading
import time
from urllib.parse import parse_qs, quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from security.userdb import check_password, hash_password, load_users  # noqa: E402

COOKIE = "mm_session"
SESSION_HOURS = 12
MAX_FAILURES = 5
LOCK_SECONDS = 60
RECHECK_SECONDS = 60
_DUMMY = hash_password("not a real password")   # so unknown names take as long as wrong passwords

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
PAGE_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; frame-ancestors 'none'; form-action 'self'; base-uri 'none'")


class Auth:
    def __init__(self, users_file, secure_cookie=True):
        self.users_file = users_file
        self.secure_cookie = secure_cookie
        self.sessions = {}          # token -> {"user", "expires", "checked"}
        self.failures = {}          # address -> {"count", "locked_until"}
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- passwords
    def check(self, user, password, address):
        """(True, None) or (False, message to show)."""
        now = time.time()
        with self.lock:
            f = self.failures.get(address)
            if f and f["locked_until"] > now:
                return False, f"Too many wrong passwords. Try again in {int(f['locked_until'] - now) + 1} s."
        stored = load_users(self.users_file).get(user)
        ok = check_password(password, stored or _DUMMY) and stored is not None
        with self.lock:
            if ok:
                self.failures.pop(address, None)
                return True, None
            f = self.failures.setdefault(address, {"count": 0, "locked_until": 0})
            f["count"] += 1
            if f["count"] >= MAX_FAILURES:
                f["count"], f["locked_until"] = 0, now + LOCK_SECONDS
                return False, f"Too many wrong passwords. Try again in {LOCK_SECONDS} s."
        return False, "Wrong name or password."

    # ---------------------------------------------------------------- sessions
    def new_session(self, user):
        token = secrets.token_urlsafe(32)
        with self.lock:
            now = time.time()
            self.sessions = {t: s for t, s in self.sessions.items() if s["expires"] > now}   # tidy up
            self.sessions[token] = {"user": user, "expires": now + SESSION_HOURS * 3600, "checked": now}
        return token

    def end_session(self, token):
        with self.lock:
            self.sessions.pop(token, None)

    def cookie_header(self, token, max_age=SESSION_HOURS * 3600):
        parts = [f"{COOKIE}={token}", "Path=/", "HttpOnly", "SameSite=Strict", f"Max-Age={max_age}"]
        if self.secure_cookie:
            parts.append("Secure")
        return "; ".join(parts)

    def user_for(self, request):
        """The signed-in user for this request, or None."""
        token = request.cookies.get(COOKIE)
        if token:
            now = time.time()
            with self.lock:
                s = self.sessions.get(token)
                if s and s["expires"] <= now:
                    self.sessions.pop(token, None)
                    s = None
            if s:
                if now - s["checked"] > RECHECK_SECONDS:            # removed from the users file?
                    if s["user"] not in load_users(self.users_file):
                        self.end_session(token)
                        return None
                    s["checked"] = now
                return s["user"]
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                user, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            except (ValueError, UnicodeDecodeError):
                return None
            ok, _ = self.check(user, password, client_address(request))
            return user if ok else None
        return None


def client_address(request):
    return request.client.host if request.client else "?"


def safe_next(target):
    """Only go back to a page on this site after signing in (no redirects to other sites)."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return "/"


def parse_form(body):
    form = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True)
    return {k: v[0] for k, v in form.items()}


def login_page(message=None, next_path="/", user=""):
    note = f'<p class="msg" role="alert">{html.escape(message)}</p>' if message else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in: Motor monitoring</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E%3Crect width=%2216%22 height=%2216%22 rx=%223%22 fill=%22%23d03b3b%22/%3E%3C/svg%3E">
<style>
:root {{ color-scheme: light dark; --page:#d9dcde; --surface:#f4f5f5; --ink:#1d2125; --ink-2:#4a525a; --rule:#c2c7cb; --crit:#d03b3b; --focus:#2a78d6;
  --display:"Bahnschrift","DIN Alternate","Barlow Semi Condensed","Roboto Condensed","Arial Narrow",sans-serif;
  --body:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root {{ --page:#16181a; --surface:#1e2124; --ink:#e7e9eb; --ink-2:#b2b8be; --rule:#363b40; --focus:#3987e5; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; min-height:100vh; display:grid; place-items:center; background:var(--page); color:var(--ink); font:15px/1.45 var(--body); }}
main {{ width:min(360px, 92vw); }}
.strip {{ display:flex; gap:4px; background:#1d2125; padding:5px; border-radius:5px 5px 0 0; }}
.strip i {{ flex:1; height:14px; background:#3a4046; border-radius:2px; }}
form {{ background:var(--surface); padding:22px 22px 18px; border-radius:0 0 6px 6px; }}
h1 {{ font:600 24px/1.1 var(--display); margin:0 0 16px; }}
label {{ display:block; font-size:13.5px; color:var(--ink-2); margin:12px 0 4px; }}
input {{ width:100%; font:inherit; padding:8px 10px; border:1px solid var(--rule); border-radius:4px; background:var(--page); color:var(--ink); }}
button {{ margin-top:18px; width:100%; font:600 15px var(--body); padding:9px; border:0; border-radius:4px; background:var(--ink); color:var(--surface); cursor:pointer; }}
:focus-visible {{ outline:3px solid var(--focus); outline-offset:2px; }}
.msg {{ margin:0 0 4px; padding:8px 10px; border-left:4px solid var(--crit); background:var(--page); border-radius:0 4px 4px 0; }}
.hint {{ font-size:13px; color:var(--ink-2); margin:14px 0 0; }}
</style></head>
<body><main>
<div class="strip" aria-hidden="true"><i></i><i></i><i></i><i></i></div>
<form method="post" action="/login">
  <h1>Motor monitoring</h1>
  {note}
  <input type="hidden" name="next" value="{html.escape(safe_next(next_path))}">
  <label for="user">Name</label>
  <input id="user" name="user" autocomplete="username" required value="{html.escape(user)}" {'' if user else 'autofocus'}>
  <label for="password">Password</label>
  <input id="password" name="password" type="password" autocomplete="current-password" required {'autofocus' if user else ''}>
  <button type="submit">Sign in</button>
  <p class="hint">The first password is in monitoring/secrets/dashboard_admin_password.txt.</p>
</form>
</main></body></html>"""


def login_redirect_target(request):
    path = request.url.path
    if request.url.query:
        path += "?" + request.url.query
    return "/login?next=" + quote(path, safe="")
