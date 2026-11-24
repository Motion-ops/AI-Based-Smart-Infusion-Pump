"""
===========================================================
WIFI_BRIDGE_TEST.PY
Smart Infusion Pump - PC <-> ESP32 wireless (TCP) test tool

Socket-based sibling of esp32_bridge.py's manual test mode, for
verifying the new WiFi control channel (firmware/pump_firmware's
WiFiServer on TCP_CONTROL_PORT) BEFORE touching native_app.py at all -
same reasoning esp32_bridge.py's "test" mode was built for on the
wired side: confirm the PC and the ESP32 can actually talk to each
other over this new transport, in isolation, before building any GUI
on top of it.

Speaks the EXACT SAME text protocol as esp32_bridge.py, over a raw TCP
socket instead of a serial port - both transports are dispatched
through the firmware's one handleSerialLine() parser, so anything that
works here should behave identically to the wired tool, including
watchdog/E-STOP/mode-rejection behavior.

Run with:
    python tools/wifi_bridge_test.py --host 192.168.1.50
    python tools/wifi_bridge_test.py --host 192.168.1.50 --port 5000
===========================================================
"""

import sys
import time
import socket
import argparse


def connect(host, port, timeout=5.0):
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.setblocking(False)
    return sock


def send_command(sock, line):
    sock.sendall((line + "\n").encode("utf-8"))


def read_available(sock, wait=0.3):
    time.sleep(wait)
    buf = b""
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
    except BlockingIOError:
        pass
    except OSError:
        pass
    return [ln for ln in buf.decode("utf-8", errors="replace").splitlines() if ln]


def print_esp32_lines(sock, wait=0.3):
    for line in read_available(sock, wait):
        print("  ESP32:", line)


def manual_test(sock):
    print()
    print("Manual wireless connectivity test.")
    print("Type a number for mL/hr, or one of:")
    print("  weight_dose <weight_kg> <dose_value> <MLHR|UGKGMIN>")
    print("  vtbi <rate_ml_hr> <vtbi_ml>")
    print("  status | stop | quit")
    print()

    while True:
        cmd = input("> ").strip()

        if cmd.lower() == "quit":
            break
        elif cmd.lower() == "status":
            send_command(sock, "GET_STATUS")
        elif cmd.lower() == "stop":
            send_command(sock, "STOP")
        elif cmd.lower().startswith("weight_dose "):
            parts = cmd.split()
            if len(parts) != 4:
                print("Usage: weight_dose <weight_kg> <dose_value> <MLHR|UGKGMIN>")
                continue
            _, weight_kg, dose_value, unit = parts
            send_command(sock, f"SET_WEIGHT_DOSE:{weight_kg},{dose_value},{unit.upper()}")
        elif cmd.lower().startswith("vtbi "):
            parts = cmd.split()
            if len(parts) != 3:
                print("Usage: vtbi <rate_ml_hr> <vtbi_ml>")
                continue
            _, rate_ml_hr, vtbi_ml = parts
            send_command(sock, f"SET_VTBI_PROGRAM:{rate_ml_hr},{vtbi_ml}")
        else:
            try:
                rate = float(cmd)
            except ValueError:
                print("Enter a number (mL/hr), 'weight_dose ...', 'vtbi ...', "
                      "'status', 'stop', or 'quit'.")
                continue
            send_command(sock, f"SET_RATE:{rate}")

        print_esp32_lines(sock)


def main():
    parser = argparse.ArgumentParser(description="ESP32 pump WiFi bridge test tool")
    parser.add_argument("--host", required=True, help="Pump's IP address, e.g. 192.168.1.50")
    parser.add_argument("--port", type=int, default=5000, help="TCP_CONTROL_PORT from the firmware")
    args = parser.parse_args()

    print(f"Connecting to {args.host}:{args.port} ...")
    try:
        sock = connect(args.host, args.port)
    except OSError as exc:
        print(f"Connect failed: {exc}")
        sys.exit(1)

    print("Connected.")
    print_esp32_lines(sock, wait=1.0)

    manual_test(sock)

    sock.close()


if __name__ == "__main__":
    main()
