"""
===========================================================
ESP32_BRIDGE.PY
Smart Infusion Pump - PC <-> ESP32 serial bridge

This is the missing link between everything built in Python this
session (digital_twin.py, expert_controller.py) and the physical
pump firmware (firmware/pump_firmware/pump_firmware.ino). It speaks
the exact Serial protocol that firmware implements:

    SET_RATE:<mL/hr>
    STOP
    GET_STATUS

Two modes:
  --mode test   Manual connectivity test - type a number to send a
                flow rate, or 'status'/'stop'/'quit'. Use this FIRST,
                before anything automated, to confirm the PC and the
                ESP32 can actually talk to each other at all.
  --mode sim    Hardware-in-the-loop: runs a REAL VirtualPatient +
                expert_controller simulation and streams the
                controller's actual computed flow rate to the
                physical pump every tick - the architecture this
                whole project has been building toward.

Run with:
    pip install pyserial
    python esp32_bridge.py                          (lists available ports)
    python esp32_bridge.py --port COM5 --mode test
    python esp32_bridge.py --port COM5 --mode sim --profile Moderate --speed 5
===========================================================
"""

import sys
import time
import argparse

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("pyserial is not installed. Run: pip install pyserial")
    sys.exit(1)


def list_available_ports():
    return list(serial.tools.list_ports.comports())


def connect(port, baud=115200, timeout=2.0):
    ser = serial.Serial(port, baud, timeout=timeout)

    # NOTE: this board uses the ESP32-S3's native USB (that's the whole
    # reason GPIO19/20 were off-limits earlier) - unlike boards with a
    # separate USB-UART bridge chip, native USB does NOT reset the chip
    # on connect. The board is very likely already running by the time
    # this script connects, so it's normal (not an error) if no boot
    # banner shows up below - it already printed before we were listening.
    time.sleep(0.5)
    ser.reset_input_buffer()

    return ser


def send_command(ser, line):
    ser.write((line + "\n").encode("utf-8"))


def read_available(ser, wait=0.3):
    time.sleep(wait)
    lines = []
    while ser.in_waiting:
        raw = ser.readline().decode("utf-8", errors="replace").strip()
        if raw:
            lines.append(raw)
    return lines


def print_esp32_lines(ser, wait=0.3):
    for line in read_available(ser, wait):
        print("  ESP32:", line)


# -----------------------------------------------------
# Manual connectivity test
# -----------------------------------------------------
def manual_test(ser):
    print()
    print("Manual connectivity test.")
    print("Type a number for mL/hr, 'CAL:<steps>' for a calibration jog,")
    print("or: status | stop | quit")
    print()

    while True:
        cmd = input("> ").strip()

        if cmd.lower() == "quit":
            break
        elif cmd.lower() == "status":
            send_command(ser, "GET_STATUS")
        elif cmd.lower() == "stop":
            send_command(ser, "STOP")
        elif cmd.upper().startswith("CAL:"):
            send_command(ser, cmd.upper())
        else:
            try:
                rate = float(cmd)
            except ValueError:
                print("Enter a number (mL/hr), 'CAL:<steps>', 'status', 'stop', or 'quit'.")
                continue
            send_command(ser, f"SET_RATE:{rate}")

        print_esp32_lines(ser)


# -----------------------------------------------------
# Hardware-in-the-loop: real digital twin + controller
# -----------------------------------------------------
def simulation_loop(ser, profile=None, seed=None, target_map=None, speed=1.0):

    from digital_twin import VirtualPatient
    from expert_controller import expert_controller

    patient = VirtualPatient(0, profile=profile, seed=seed, target_map=target_map)

    print(f"Simulating {patient.profile} patient "
          f"(SAPS II {patient.saps_ii}, target {patient.target_bp:.0f} mmHg).")
    print("Sending real controller-computed flow rate to the physical pump every tick.")
    print("Ctrl+C to stop (sends STOP to the pump before exiting).")
    print()

    try:
        while True:
            patient.update()
            decision = expert_controller(patient)
            patient.flow_rate = decision["RecommendedFlow"]

            send_command(ser, f"SET_RATE:{patient.flow_rate:.2f}")

            print(f"t={len(str(patient))!s:0s}"[:0], end="")  # no-op, keeps line width stable
            print(f"MAP={patient.map:6.1f}  HR={patient.hr:6.1f}  "
                  f"flow={patient.flow_rate:6.1f} mL/hr  mode={decision['Mode']:10s}", end="")

            esp_lines = read_available(ser, wait=0.05)
            if esp_lines:
                print("  <-", esp_lines[-1])
            else:
                print()

            time.sleep(1.0 / speed)

    except KeyboardInterrupt:
        print()
        print("Stopping - sending STOP to the pump.")
        send_command(ser, "STOP")
        print_esp32_lines(ser, wait=0.5)


# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ESP32 pump serial bridge")
    parser.add_argument("--port", help="Serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mode", choices=["test", "sim"], default="test")
    parser.add_argument("--profile", default=None,
                         help="Healthy/Mild/Moderate/Severe - omit for random")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target", type=float, default=None,
                         help="Target MAP mmHg, 65-85 - omit for default (65.0)")
    parser.add_argument("--speed", type=float, default=1.0,
                         help="Simulation speed multiplier (ticks/sec)")
    args = parser.parse_args()

    if not args.port:
        ports = list_available_ports()
        print("No --port given. Available serial ports:")
        if not ports:
            print("  (none found - is the board plugged in?)")
        for p in ports:
            print(f"  {p.device}  -  {p.description}")
        print()
        print("Re-run with --port <name>, e.g.:")
        print("  python esp32_bridge.py --port COM5 --mode test")
        sys.exit(1)

    print(f"Connecting to {args.port} at {args.baud} baud...")
    ser = connect(args.port, args.baud)
    print("Connected.")
    print_esp32_lines(ser, wait=1.0)

    if args.mode == "test":
        manual_test(ser)
    else:
        simulation_loop(
            ser,
            profile=args.profile,
            seed=args.seed,
            target_map=args.target,
            speed=args.speed,
        )

    ser.close()


if __name__ == "__main__":
    main()
