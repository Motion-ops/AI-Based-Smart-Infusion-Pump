"""
===========================================================
CLINICIAN_DASHBOARD.PY
Smart AI Infusion Pump - Read-only phone/browser vitals dashboard

A SEPARATE process from native_app.py, deliberately - keeps that app's
zero-asyncio, QTimer-polled simplicity intact rather than folding a
WebSocket server into the Qt event loop.

Data flow (view-only - there is NO command path back to the pump from
here, by design, same reasoning as the desktop app's own wired/
wireless control staying the single authorized channel):

    native_app.py's _on_timer() --[UDP, localhost, fire-and-forget]-->
    this process's UDP listener thread --[WebSocket broadcast]-->
    any number of phone/browser tabs open to this page

native_app.py never blocks on this - UDP send is fire-and-forget, and
if this process isn't even running, those sends just go nowhere with
no error. This process never talks to the pump directly, wired or
wireless - it only ever receives what native_app.py chooses to send.

Run with:
    pip install flask flask-sock
    python clinician_dashboard.py
Then, from a phone/browser on the SAME network as this PC:
    http://<this-PC's-LAN-IP>:8765
===========================================================
"""

import json
import socket
import threading

from flask import Flask, Response
from flask_sock import Sock

UDP_LISTEN_HOST = "127.0.0.1"   # native_app.py only ever sends from this same PC
UDP_LISTEN_PORT = 9999
HTTP_HOST = "0.0.0.0"           # reachable from phones on the LAN/hotspot
HTTP_PORT = 8765

app = Flask(__name__)
sock = Sock(app)

_clients = set()
_clients_lock = threading.Lock()
_latest_snapshot = None
_latest_lock = threading.Lock()


def _broadcast(snapshot):
    global _latest_snapshot
    with _latest_lock:
        _latest_snapshot = snapshot

    payload = json.dumps(snapshot)
    with _clients_lock:
        dead = []
        for ws in _clients:
            try:
                ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _clients.discard(ws)


def _udp_listener():
    """Runs forever in its own daemon thread. Never touches the pump -
    only ever receives whatever native_app.py chose to send."""
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((UDP_LISTEN_HOST, UDP_LISTEN_PORT))
    print(f"[dashboard] UDP listener on {UDP_LISTEN_HOST}:{UDP_LISTEN_PORT}")

    while True:
        data, _addr = udp_sock.recvfrom(65536)
        try:
            snapshot = json.loads(data.decode("utf-8"))
        except Exception:
            continue
        _broadcast(snapshot)


@app.route("/")
def index():
    return Response(_PAGE_HTML, mimetype="text/html")


@sock.route("/ws")
def ws_route(ws):
    with _clients_lock:
        _clients.add(ws)

    # A just-joined viewer gets the current snapshot immediately,
    # rather than waiting for the next tick from native_app.py.
    with _latest_lock:
        if _latest_snapshot is not None:
            try:
                ws.send(json.dumps(_latest_snapshot))
            except Exception:
                pass

    try:
        while True:
            # This is a view-only dashboard - we never expect the
            # client to send anything. The only purpose of this
            # blocking receive() is to detect disconnection (returns
            # None when the socket closes), while _broadcast() above
            # pushes data to this same connection object from the
            # separate UDP-listener thread.
            msg = ws.receive()
            if msg is None:
                break
    finally:
        with _clients_lock:
            _clients.discard(ws)


_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pump Vitals</title>
<style>
  :root {
    --bg: #0b0f14; --card: #131a22; --border: #223040;
    --text: #e6edf3; --dim: #7d8a99;
    --ok: #35c46a; --warn: #e6b800; --crit: #e04b4b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  h1 {
    font-size: 18px; font-weight: 600; margin: 0 0 4px 0;
  }
  #status { font-size: 13px; color: var(--dim); margin-bottom: 16px; }
  #status.live { color: var(--ok); }
  #status.stale { color: var(--warn); }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
  }
  .label { font-size: 12px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.04em; }
  .value { font-size: 28px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .unit { font-size: 13px; color: var(--dim); margin-left: 4px; font-weight: 400; }
  .value.ok { color: var(--ok); }
  .value.warn { color: var(--warn); }
  .value.crit { color: var(--crit); }
  .wide { grid-column: 1 / -1; }
  .mode-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
  .pill {
    background: var(--card); border: 1px solid var(--border); border-radius: 999px;
    padding: 6px 12px; font-size: 13px; color: var(--dim);
  }
  .pill b { color: var(--text); }
</style>
</head>
<body>
  <h1>Smart Infusion Pump - Live Vitals</h1>
  <div id="status">Connecting...</div>

  <div class="grid">
    <div class="card">
      <div class="label">MAP</div>
      <div class="value" id="v-map">--<span class="unit">mmHg</span></div>
    </div>
    <div class="card">
      <div class="label">Heart Rate</div>
      <div class="value" id="v-hr">--<span class="unit">bpm</span></div>
    </div>
    <div class="card">
      <div class="label">SpO2</div>
      <div class="value" id="v-spo2">--<span class="unit">%</span></div>
    </div>
    <div class="card">
      <div class="label">Flow Rate</div>
      <div class="value" id="v-flow">--<span class="unit">mL/hr</span></div>
    </div>
  </div>

  <div class="mode-row">
    <div class="pill">Mode: <b id="v-mode">--</b></div>
    <div class="pill">Target MAP: <b id="v-target">--</b></div>
    <div class="pill">Risk score: <b id="v-risk">--</b></div>
  </div>

  <script>
    const statusEl = document.getElementById("status");
    const els = {
      map: document.getElementById("v-map"),
      hr: document.getElementById("v-hr"),
      spo2: document.getElementById("v-spo2"),
      flow: document.getElementById("v-flow"),
      mode: document.getElementById("v-mode"),
      target: document.getElementById("v-target"),
      risk: document.getElementById("v-risk"),
    };

    let lastMessageAt = 0;

    function setValue(el, value, unit, cls) {
      el.innerHTML = (value === undefined || value === null ? "--" : value)
        + '<span class="unit">' + unit + '</span>';
      el.className = "value" + (cls ? " " + cls : "");
    }

    function classifyMap(map, target) {
      if (map === undefined || map === null) return "";
      if (target && Math.abs(map - target) <= 3) return "ok";
      if (map < 50 || map > 120) return "crit";
      return "warn";
    }

    function classifySpo2(spo2) {
      if (spo2 === undefined || spo2 === null) return "";
      if (spo2 >= 95) return "ok";
      if (spo2 >= 90) return "warn";
      return "crit";
    }

    function render(d) {
      lastMessageAt = Date.now();
      setValue(els.map, d.MAP, "mmHg", classifyMap(d.MAP, d.TargetBP));
      setValue(els.hr, d.HR, "bpm", "");
      setValue(els.spo2, d.SpO2, "%", classifySpo2(d.SpO2));
      setValue(els.flow, d.ControllerFlow ?? d.FlowRate, "mL/hr", "");
      els.mode.textContent = d.ControllerMode ?? d.Mode ?? "--";
      els.target.textContent = d.TargetBP !== undefined ? d.TargetBP + " mmHg" : "--";
      els.risk.textContent = d.RiskScore !== undefined ? (d.RiskScore * 100).toFixed(1) + "%" : "--";
    }

    function connect() {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(proto + "//" + location.host + "/ws");

      ws.onopen = () => {
        statusEl.textContent = "Connected - waiting for data...";
        statusEl.className = "live";
      };
      ws.onmessage = (ev) => {
        try { render(JSON.parse(ev.data)); } catch (e) { /* ignore malformed */ }
        statusEl.textContent = "Live";
        statusEl.className = "live";
      };
      ws.onclose = () => {
        statusEl.textContent = "Disconnected - retrying...";
        statusEl.className = "stale";
        setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
    }

    setInterval(() => {
      if (lastMessageAt && Date.now() - lastMessageAt > 5000) {
        statusEl.textContent = "No data for a few seconds - pump app may be paused/disconnected";
        statusEl.className = "stale";
      }
    }, 2000);

    connect();
  </script>
</body>
</html>
"""


def _get_lan_ip():
    """[FIX, 2026-10-12] socket.gethostbyname(socket.gethostname()) is
    unreliable on Windows the instant there's more than one network
    adapter (WiFi + Ethernet + any VPN) - it can return a completely
    wrong interface's address (e.g. a VPN's internal IP, or even
    127.0.0.1), which would have a phone trying to reach an address
    that was never actually reachable, looking indistinguishable from
    a firewall block. This instead asks the OS which LOCAL address it
    would use to reach the outside world (a real route lookup - no
    packet actually leaves, and the destination doesn't need to be
    reachable) which reliably resolves to the real WiFi-facing IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    listener = threading.Thread(target=_udp_listener, daemon=True)
    listener.start()

    local_ip = _get_lan_ip()
    print(f"[dashboard] Open from a phone/browser on this network: http://{local_ip}:{HTTP_PORT}")
    print(f"[dashboard] Or locally: http://127.0.0.1:{HTTP_PORT}")
    print("[dashboard] If the phone can't connect, check Windows Firewall allowed Python")
    print("[dashboard] to accept connections on private networks (a popup may have appeared")
    print("[dashboard] the first time this ran - if dismissed/blocked, phones can't reach it).")

    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False)


if __name__ == "__main__":
    main()
