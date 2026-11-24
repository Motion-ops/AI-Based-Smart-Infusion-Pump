#include <Keypad.h>
#include <Wire.h>
#include <LiquidCrystal_PCF8574.h>
#include <AccelStepper.h>
#include <Preferences.h>
#include <WiFi.h>

/*
  ============================================================
  SMART SYRINGE INFUSION PUMP
  REAL PUMP CONTROL FIRMWARE - V1
  ============================================================

  BOARD:
    ESP32-S3

  This is the REAL firmware, not the connection-test diagnostic
  sketch. It runs the motor CONTINUOUSLY at a commanded flow
  rate, sent by the PC over Serial - it does not jog fixed step
  counts like the test sketch does.

  SERIAL PROTOCOL (from the PC, one line at a time - identical whether
  it arrives over wired Serial or the wireless TCP control channel,
  see WIRELESS CONTROL below):
    SET_RATE:<mL/hr>   e.g. "SET_RATE:12.5"
                       Clamped to [0, MAX_FLOW_ML_HR]. Resets
                       the watchdog timer.
    SET_WEIGHT_DOSE:<weight_kg>,<dose_value>,<unit>
                       e.g. "SET_WEIGHT_DOSE:72.5,0.05,UGKGMIN" or
                       "SET_WEIGHT_DOSE:72.5,12.5,MLHR". unit is
                       MLHR or UGKGMIN. One atomic command - the PC-
                       facing equivalent of the keypad's weight-dose
                       program (A -> 2), but sent fully-formed instead
                       of digit-by-digit. Resets the watchdog timer
                       (unlike the keypad path, this does NOT set
                       manualModeActive - a remote command stays
                       watchdog-covered, same as SET_RATE).
    SET_VTBI_PROGRAM:<rate_ml_hr>,<vtbi_ml>
                       e.g. "SET_VTBI_PROGRAM:50.0,100.0". One atomic
                       command - the PC-facing equivalent of the
                       keypad's Rate+VTBI program (A -> 1), but sent
                       fully-formed instead of rate-then-VTBI in two
                       separate keystroke steps. Same watchdog
                       behavior as SET_WEIGHT_DOSE above.
    STOP               Immediately zero the rate and disable
                       the driver.
    GET_STATUS         Prints one status line (see printStatusLine()).
    CAL:<steps>        Calibration jog - runs exactly <steps> step
                       pulses forward at a fixed slow speed, then
                       stops. Only when idle (not mid-run), not
                       watchdog-guarded. See CALIBRATION note below.
    RETRACT            Syringe reload jog - reverses toward HOME at a
                       fixed slow speed, stops automatically at the
                       HOME limit (or a step-count safety ceiling if
                       that switch never triggers). Only when idle.

  WIRELESS CONTROL:
    The ESP32-S3 joins a WiFi network (STA mode, credentials in
    WIFI_SSID/WIFI_PASSWORD below) and listens on TCP_CONTROL_PORT for
    ONE wireless control client at a time - a second simultaneous
    connection attempt is refused outright, so the pump can never have
    two different senders issuing conflicting commands at once. That
    one client uses the EXACT SAME text-line protocol as wired Serial,
    dispatched through the same handleSerialLine() - wired and wireless
    are two transports feeding one command parser, not two separate
    protocols. A failed/missing WiFi connection never blocks wired or
    keypad operation - see setupWifi()/maintainWifi().

  WATCHDOG:
    If no SET_RATE/SET_WEIGHT_DOSE/SET_VTBI_PROGRAM command arrives
    within WATCHDOG_TIMEOUT_MS, the pump automatically stops -
    protects against a PC crash, USB disconnect, dropped WiFi link, or
    a hung control loop upstream.

  KEYPAD (manual override, available even during automated run):
    A = start manual entry - first choose a program:
          1 = Rate + VTBI:      rate (mL/hr) -> VTBI (mL) -> runs at
                                that rate, auto-stops when delivered
          2 = Weight-based dose: weight -> unit -> rate/dose, every
                                time - nothing remembered between entries
          3 = Volume + Time:    volume (mL) -> time (min) -> pump
                                computes rate=volume/time itself, then
                                runs it and auto-stops when delivered
                                (same VTBI auto-stop tracking as #1)
        during entry: 0-9 = digit, * = decimal point,
                      # = confirm/next, D = cancel entire entry,
                      C = backspace (delete last digit typed - fixes a
                          mistake without cancelling and starting over),
                      B/C = select mL/hr or ug/kg/min (weight-dose unit
                            step only - C means "backspace" everywhere
                            else, since that step has no digits to edit)
    R = reset E-STOP (after physically releasing it)
    T = print status
    V / * = show battery on LCD (idle only - * means decimal point
            while a dose entry is in progress)
    # = soft-stop (idle only - same as sending STOP over Serial)
    C = retract for syringe reload (idle only - reverses toward HOME,
        stops automatically; same key means "backspace" mid-entry,
        see above - the two never overlap since this only fires when
        entryMode is NONE)
    B = open SETTINGS menu (idle only - rejected while pumpShouldRun,
        same reasoning as C/0):
          1 = toggle sound ON/OFF (persisted across reboots via NVS/
              Preferences, key "pump"/"sound" - mutes routine keypad/
              UI feedback beeps only (digit entry, invalid-entry
              retries, menu navigation, rejects). Safety-critical
              beeps (E-STOP trigger/reset, watchdog stop, end-limit
              stop, VTBI complete) and the boot-ready chime always
              sound regardless of this setting - see beep()'s
              alwaysAudible parameter)
          2 = toggle WiFi ON/OFF (persisted across reboots via NVS/
              Preferences, key "pump"/"wifi" - takes effect on the
              NEXT boot, not immediately. Added after confirming WiFi's
              TX current draw was the cause of a battery-only reboot
              loop - switch OFF for battery-only use until the
              hardware fix, in place, wired USB use is unaffected
              either way)
          3 = retract for syringe reload (same action as the standalone
              C shortcut above - offered here too so it doesn't have to
              be memorized as a raw key)
          D or 0 = back to the idle screen

  SAFETY (all reused, unchanged in behavior, from the proven
  connection-test sketch):
    - Driver starts DISABLED.
    - E-STOP disables the driver and latches until reset.
    - HOME limit blocks/stops reverse movement.
    - END limit blocks/stops forward movement.
    - Rate is always clamped to MAX_FLOW_ML_HR - never sent to
      the motor above that, regardless of what the PC requests.

  CALIBRATION - VERIFY BEFORE TRUSTING DELIVERED VOLUME:
    STEPS_PER_ML below is derived from the syringe/lead-screw
    geometry in the design calculations (19.13mm bore, 8mm lead
    screw pitch, 3200 microsteps/turn assuming 1/16 microstepping).
    The 1/16 setting depends on how MS1/MS2/MS3 are physically
    wired on your specific A4988 module - this firmware has no
    control over that (those pins aren't wired to the ESP32 at
    all per the pin allocation). CONFIRM your board's actual
    microstepping setting before trusting any delivered-volume
    number - if it's wired differently, command 3200 steps and
    measure whether the plunger actually travels 8mm; recalculate
    STEPS_PER_ML from the real result if it doesn't.

    Use "CAL:3200" over Serial to do this (see startCalibration()) -
    it runs exactly that many step pulses forward at a fixed slow
    speed and reports the expected mL when done, so you can measure
    the real travel/volume and compare.

  IMPORTANT:
    Do NOT connect or disconnect the NEMA 17 while the A4988 is
    powered. This is bench-test firmware for a research/simulation
    prototype - NOT clinical infusion-pump firmware.

  [FIX, 2026-10-01] Serial -> Serial. This board's native USB-CDC
  Serial interface was confirmed non-functional via isolated testing
  (a minimal sketch printing to both Serial and Serial showed
  "UART0 SERIAL" repeating forever over COM9, but "USB SERIAL" never
  appeared at all - with USB CDC On Boot already Enabled and USB Mode
  already Hardware CDC and JTAG, ruling out the usual config culprits).
  Whatever the underlying cause (this board's specific USB silicon/
  wiring, or a driver quirk), native CDC does not work on this unit,
  while UART0 (Serial) does. Every Serial.* call in this file was
  switched to Serial.* so the SAME physical COM port (whatever
  enumerates UART0 - directly, or via an onboard bridge chip) that
  already proved itself working carries all PC communication: boot
  banner, GET_STATUS, and - critically - incoming SET_RATE/STOP/etc.
  commands, which is what native_app.py's PumpLink actually depends on.
  ============================================================
*/


// ============================================================
// PIN DEFINITIONS (proven in connection_test.ino - unchanged)
// ============================================================

// ---------------- KEYPAD ----------------

#define ROW4 4
#define ROW3 5
#define ROW2 6
#define ROW1 7

#define COL4 15
#define COL3 16
#define COL2 17
#define COL1 18


// ---------------- A4988 ----------------

#define STEP_PIN 2
#define DIR_PIN  21
#define EN_PIN   42


// ---------------- LCD ----------------

#define SDA_PIN 8
#define SCL_PIN 9


// ---------------- SAFETY / OUTPUT ----------------

#define E_STOP_PIN     1
#define HOME_LIMIT_PIN 10
#define END_LIMIT_PIN  11
#define BUZZER_PIN     12


// ---------------- STATUS LEDS ----------------

#define GREEN_LED_PIN 13
#define RED_LED_PIN   14

#define RED_FLASH_INTERVAL_MS 300


// ---------------- BATTERY VOLTAGE DIVIDER ----------------
// See connection_test.ino for the full GPIO3-strapping-pin caution -
// same wiring, same caveat, already boot-tested there.

#define BATTERY_ADC_PIN 3

#define DIVIDER_R1_OHMS 100000.0
#define DIVIDER_R2_OHMS 27000.0   // [2026-10-01] confirmed installed - was previously a TODO placeholder

// [ADDED 2026-10-01] Nominal (rated) resistor values above are never
// exactly what's physically installed - ordinary 1-5% component
// tolerance alone is enough to shift the computed battery voltage by
// a few percent, which is a few percentage points of displayed
// battery level too (the voltage-to-percent curve below is fairly
// steep in the 11-12V range). This multiplies the final computed
// voltage to correct for that, without needing to re-measure or
// re-derive the actual resistor values.
//
// HOW TO CALIBRATE: get ONE known-accurate reference voltage for the
// battery at rest (a multimeter is easiest; a fully-charged pack's
// rated float voltage also works in a pinch) at the same moment you
// read this firmware's own measurement (GET_STATUS's battery_v field
// over Serial, or the LCD battery screen). Then:
//     BATTERY_VOLTAGE_CALIBRATION_FACTOR = true_voltage / firmware_reported_voltage
// Example: multimeter reads 11.40V, firmware reports 11.02V ->
// factor = 11.40 / 11.02 = 1.034. Leave at 1.0 until you have a real
// reference reading - guessing a factor with no data point would just
// trade one unverified number for another.
// [CALIBRATED 2026-11-11] Multimeter read 11.40V at rest while the
// firmware's own GET_STATUS/LCD reported 10.30V at the same moment ->
// factor = 11.40 / 10.30 = 1.107.
#define BATTERY_VOLTAGE_CALIBRATION_FACTOR 1.107

#define ADC_VREF 3.3
#define ADC_MAX_COUNT 4095.0

#define BATTERY_REPORT_INTERVAL_MS 5000
#define BATTERY_EMA_ALPHA 0.2


// ============================================================
// WIRELESS CONTROL (WiFi)
// ============================================================
// [ADDED 2026-10-07] STA mode - this board JOINS a network you
// control (e.g. a phone hotspot at a defense/demo) rather than
// hosting its own Access Point. Deliberate: an AP-mode ESP32 has no
// internet uplink, so it couldn't publish to Adafruit IO directly;
// STA mode puts the pump, the PC, and any phone/clinician dashboard
// viewers on one shared network with internet access, for both
// control and telemetry. Fill in your actual network credentials
// before flashing - these are placeholders.
//
// [2026-10-10] Confirmed by direct A/B test on the bench: WiFi's TX
// current draw was the actual cause of a battery-only reboot loop
// (fine on USB, looped on battery) - even with WIFI_TX_POWER below
// capped. Whether WiFi actually runs is now the persisted runtime
// `wifiEnabled` setting (Settings menu 'B' -> '2'), not a compile-time
// flag - see wifiEnabled's own comment near the other NVS-persisted
// settings. Leave it ON for USB/laptop-powered use (today's case);
// switch it OFF from the keypad for any battery-only run until the
// hardware fix (bulk capacitor at the ESP32's own 3.3V rail, see
// WIFI_TX_POWER's comment below) is actually in place - no re-flash
// needed either way.
#define WIFI_SSID "Infinix NOTE 10"
#define WIFI_PASSWORD "motion2239"
#define TCP_CONTROL_PORT 5000
#define WIFI_CONNECT_TIMEOUT_MS 10000
#define WIFI_RECONNECT_CHECK_INTERVAL_MS 5000

// [ADDED 2026-10-10] WiFi's radio draws current in short, sharp TX
// bursts (full power can spike to 300-500mA for tens of ms) - a USB
// port absorbs that fine, but a battery pack's regulator/charge
// module is usually sized for the ESP32's much lower AVERAGE draw,
// not that peak. On this board that spike was enough to sag the rail
// below the brownout threshold and reset the chip - repeatedly,
// exactly in sync with WiFi's TX bursts, which is why it only showed
// up on battery, never on USB, and only after WiFi was added. Capping
// transmit power directly caps that peak current draw. 8.5dBm is a
// conservative starting point (default/max is 19.5dBm) - plenty for a
// benchtop/demo setup where the pump and the hotspot are in the same
// room; raise it later ONLY if you also upgrade the power path (see
// setupWifi()'s comment) and confirm on the bench it's still stable.
#define WIFI_TX_POWER WIFI_POWER_8_5dBm


// ============================================================
// PUMP / MOTOR SETTINGS
// ============================================================

// A4988 enable logic - LOW = enabled, HIGH = disabled
#define DRIVER_ENABLED  LOW
#define DRIVER_DISABLED HIGH

// [DERIVED] Same number used throughout the Python digital twin
// (MAX_FLOW_ML_HR) - top row of Medinfo Galway's Table 3 peripheral
// dosing chart (0.5 mcg/kg/min at 100 kg = 188 mL/hr, rounded up),
// NOT an explicitly stated hard cap in that guideline (corrected
// 2026-10-05 - see REFERENCES.md).
#define MAX_FLOW_ML_HR 190.0
#define MIN_FLOW_ML_HR 0.0

// --- Syringe / lead-screw geometry -> steps-per-mL derivation ---
// See the "CALIBRATION" note in the header comment before trusting this.
#define SYRINGE_BORE_DIAMETER_MM 19.13
#define LEAD_SCREW_PITCH_MM_PER_TURN 8.0
#define MICROSTEPS_PER_TURN 3200.0   // 200 full steps/rev x 1/16 microstepping

const float SYRINGE_BORE_AREA_MM2 =
  3.14159265 * (SYRINGE_BORE_DIAMETER_MM / 2.0) * (SYRINGE_BORE_DIAMETER_MM / 2.0);

const float ML_PER_TURN =
  (SYRINGE_BORE_AREA_MM2 * LEAD_SCREW_PITCH_MM_PER_TURN) / 1000.0;   // mm^3 -> mL

// [CALIBRATED 2026-11-11] Iterative bench correction - delivered volume
// scales directly with this factor (it scales step rate, not the VTBI
// stop-time accounting), so each pass multiplies the prior factor by
// target/actual:
//   Pass 1: factor=1.0   -> commanded 5.00 mL, actual 4.50 mL (0.9x)
//           next factor = 1.0 * 5.00/4.50 = 1.111
//   Pass 2: factor=1.111 -> commanded 5.00 mL, actual 5.20 mL (1.04x, overshot)
//           next factor = 1.111 * 5.00/5.20 = 1.068
//   Pass 3: factor=1.068 -> commanded 5.00 mL, actual 5.08 mL (1.016x),
//           repeated once more with the same 5.08 mL result - a
//           consistent, repeatable overshoot, not measurement noise.
//           next factor = 1.068 * 5.00/5.08 = 1.051
//   Pass 4: factor=1.051 -> commanded 5.00 mL, actual 5.20 mL (1.04x).
//           [CAUTION] This is BACKWARDS from pass 3 - a LOWER factor
//           produced a HIGHER actual volume, which the proportional
//           model (volume scales directly with this factor) cannot
//           explain. Dose-based measurement noise (syringe-reading
//           precision, priming/backlash variability between runs) is
//           now comparable in size to the correction steps themselves.
//           Applying the same formula anyway: next factor =
//           1.051 * 5.00/5.20 = 1.011 - but if the NEXT real dose test
//           still bounces around rather than converging, stop
//           iterating on whole-dose tests and switch to the CAL: jog
//           test (see header "CALIBRATION" note) instead - it isolates
//           pure step count vs. measured plunger travel, with no
//           syringe-reading or priming variability in the loop.
// Passes 2-4 above turned out to be measuring noise, not a real signal
// - likely because the board wasn't actually running the edited code
// for several of those passes (a recurring stale-upload issue this
// session - the Arduino IDE compiles from its own open editor buffer,
// not the file on disk, so an externally-edited .ino needs the sketch
// closed and reopened before the next Upload or it silently reflashes
// the old value). Once that was accounted for, a much larger,
// unambiguous step - factor=0.9 - gave a correct 5.00 mL result on a
// commanded 5.00 mL dose and is confirmed working. This is a stopgap
// from dose measurements (syringe-reading error included); re-verify
// with the CAL: jog test (see header "CALIBRATION" note) if tighter
// accuracy is ever needed, and keep iterating (factor *= target/
// actual) only after confirming via Serial's own boot-time "Derived
// STEPS_PER_ML" printout that the board is actually running the code
// you just edited.
#define VOLUME_CALIBRATION_FACTOR 0.9

const float STEPS_PER_ML = (MICROSTEPS_PER_TURN / ML_PER_TURN) * VOLUME_CALIBRATION_FACTOR;

// Motion smoothness - acceleration when the commanded rate changes.
// Not a safety limit (MAX_FLOW_ML_HR is) - just avoids a mechanically
// abrupt jump in speed every time SET_RATE changes.
#define MOTOR_ACCELERATION_STEPS_S2 200.0

// If no valid SET_RATE arrives within this long, stop automatically.
#define WATCHDOG_TIMEOUT_MS 3000

// [CITED] Same value as Python constants.py's DRUG_CONCENTRATION_MCG_ML -
// standard premix concentration (4 mg norepinephrine in 250 mL D5W).
// mL/hr = (mcg/kg/min * weight_kg * 60) / DRUG_CONCENTRATION_MCG_ML
#define DRUG_CONCENTRATION_MCG_ML 16.0

// Fixed, deliberately slow/safe speed used ONLY by the CAL: calibration
// jog (see "CALIBRATION" note above). Doesn't affect accuracy - accuracy
// comes from the exact step count you command and physically measure -
// this just controls how long the test takes to run.
#define CALIBRATION_SPEED_ML_HR 150.0

// Fixed speed used ONLY by the RETRACT syringe-reload jog. Same
// reasoning as CALIBRATION_SPEED_ML_HR - this is a deliberate bench
// action, not a dosing rate, so it just needs to be a speed an
// operator can watch and stop safely, not clinically meaningful.
// RETRACT_MAX_STEPS below is a fixed STEP/distance ceiling, not a
// time-based one, so it does not need to change when this speed
// changes: the worst-case travel distance if HOME_LIMIT_PIN fails is
// identical either way, only the TIME to reach it shrinks, giving an
// operator watching it slightly less reaction time in that specific
// failure case - acceptable since the step ceiling still bounds it to
// one lead-screw length regardless.
// [UPDATED 2026-09-30, was 150.0 then 180.0] Raised to 250.0, now
// ABOVE MAX_FLOW_ML_HR=190.0 at the user's explicit request for a
// faster reload. This is now the first speed in this firmware to
// exceed the pump's dosing-rate ceiling - stepper.setMaxSpeed() (see
// setup()) is derived from max(MAX_FLOW_ML_HR, RETRACT_SPEED_ML_HR) so
// this speed is actually reachable rather than silently clamped by
// AccelStepper's own internal constrain() in setSpeed(). Actual dosing
// stays independently capped at MAX_FLOW_ML_HR regardless (applyRate()
// clamps before ever calling setSpeed() - see there), so this does NOT
// raise the effective dosing-rate limit, only the retract jog's speed.
// [CAUTION] 250 mL/hr-equivalent step rate has not been validated on
// the real motor/driver/microstepping setup - test on the bench (watch
// and listen for stalling/skipped steps during a real retract) before
// trusting this for unattended use.
#define RETRACT_SPEED_ML_HR 250.0

// Backup safety ceiling on the RETRACT jog, in steps, in case
// HOME_LIMIT_PIN is disconnected, misaligned, or otherwise fails to
// trigger - without this, a faulty switch would let the carriage drive
// itself off the end of the lead screw. This is a fixed DISTANCE
// (roughly a full lead-screw length of travel), independent of
// RETRACT_SPEED_ML_HR - tune if your rail is longer than that.
#define RETRACT_MAX_STEPS 20000L


// ============================================================
// KEYPAD
// ============================================================

const byte ROWS = 4;
const byte COLS = 4;

char keys[ROWS][COLS] = {
  {'1', '2', '3', 'A'},
  {'4', '5', '6', 'B'},
  {'7', '8', '9', 'C'},
  {'*', '0', '#', 'D'}
};

// [FIXED 2026-11-06] Inverted row and column pin order to correct for
// the 180-degree reversed keypad connector wiring (where * read as A, 6 as 8,
// 5 as 9, and 1 as D/back).
byte rowPins[ROWS] = { ROW4, ROW3, ROW2, ROW1 };
byte colPins[COLS] = { COL4, COL3, COL2, COL1 };

Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);


// ============================================================
// LCD
// ============================================================

LiquidCrystal_PCF8574 lcd(0x27);


// ============================================================
// STEPPER
// ============================================================

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);


// ============================================================
// GLOBAL STATE
// ============================================================

bool emergencyStopLatched = false;
bool driverEnabled = false;

float commandedRateMLHr = 0.0;
unsigned long lastRateCommandTime = 0;
bool pumpShouldRun = false;

// -----------------------------------------------------
// OPERATING MODE - chosen explicitly at boot (or re-selected from idle
// via '0'), rather than implicitly by "whoever sent a command last."
// MANUAL: keypad dose entry (A) works, PC SET_RATE is rejected.
// AUTONOMOUS: PC SET_RATE works, keypad dose entry (A) is rejected.
// R (E-STOP reset), T (status), V/* (battery), # (soft-stop) work in
// either mode - those are universal safety/utility functions, not
// control-source-specific.
// -----------------------------------------------------
enum OperatingMode {
  MODE_UNSELECTED,
  MODE_MANUAL,
  MODE_AUTONOMOUS
};

OperatingMode operatingMode = MODE_UNSELECTED;

bool redLedState = false;
unsigned long lastRedToggle = 0;

unsigned long lastBatteryReport = 0;
float smoothedBatteryVoltage = -1.0;

String serialLineBuffer = "";

// The Keypad library's getKey() returns the SAME character on every
// scan for as long as a button stays physically down (its internal
// state machine reports both PRESSED and HOLD, not just the initial
// press edge) - without tracking this, a single normal press-and-release
// fires every handler (several of which beep(), which blocks) many
// times in a row, which is what was actually causing the multi-second
// "freeze" before the screen updated - not a delay anywhere, a flood of
// repeated redraws/beeps stacking up. This returns a key only once per
// physical press.
char lastKeyPressed = NO_KEY;

char getNewKeyPress() {

  char key = keypad.getKey();

  if (key == NO_KEY) {
    lastKeyPressed = NO_KEY;   // released - next press of anything is "new" again
    return NO_KEY;
  }

  if (key == lastKeyPressed) {
    return NO_KEY;   // still holding the same key down - not a new press
  }

  lastKeyPressed = key;
  return key;
}

// -----------------------------------------------------
// MANUAL DOSE ENTRY (keypad) - Option B: weight AND rate are entered
// together every time, nothing is carried over from a previous entry.
// This is deliberate - reusing a stale weight from the last patient to
// compute a dose would be a real safety issue, not just an inconvenience.
// -----------------------------------------------------
// Which manual program is active - scaffolding for the eventual
// Bolus / KVO programs discussed, but only Rate+VTBI, Volume+Time, and
// the pre-existing weight-based dose are actually built right now.
enum ManualProgram {
  PROGRAM_NONE,
  PROGRAM_RATE_VTBI,
  PROGRAM_WEIGHT_DOSE,
  PROGRAM_VOLUME_TIME
};

ManualProgram selectedProgram = PROGRAM_NONE;

enum EntryMode {
  ENTRY_NONE,
  ENTRY_PROGRAM_SELECT,   // "1=Rate+VTBI  2=Weight dose  3=Volume+Time"
  ENTRY_WEIGHT,           // weight-dose program only
  ENTRY_UNIT_SELECT,      // weight-dose program only
  ENTRY_RATE,             // rate-vtbi and weight-dose - rate-vtbi enters
                          // this first; weight-dose reaches it after unit select
  ENTRY_VTBI,             // rate-vtbi program only, after rate is confirmed
  ENTRY_VT_VOLUME,        // volume-time program only - volume, entered first
  ENTRY_VT_TIME           // volume-time program only - time, after volume is confirmed
};

EntryMode entryMode = ENTRY_NONE;
String entryBuffer = "";
float enteredWeightKg = 0.0;
bool enteredUnitIsMcgKgMin = false;
float pendingRateMLHr = 0.0;   // rate-vtbi program: held here between the rate and VTBI steps
float pendingVolumeML = 0.0;   // volume-time program: held here between the volume and time steps

// True whenever the CURRENT running rate was set from the keypad, not
// the PC - the PC-communication watchdog is meaningless in this state
// (there's no PC in the loop to go quiet), so it's skipped while this
// is true. Any SET_RATE arriving from the PC clears it immediately -
// the PC always takes back control the instant it sends a command.
bool manualModeActive = false;

// -----------------------------------------------------
// VTBI (volume to be infused) - Rate+VTBI program only. Tracks
// delivered volume from the commanded rate and elapsed time, the same
// approach already used and validated in the Python digital twin
// (VirtualPatient.update_delivered_volume()), just ported to the
// firmware. Auto-stops the pump the instant the target is reached.
// -----------------------------------------------------
bool vtbiActive = false;
float vtbiTargetML = 0.0;
float deliveredVolumeML = 0.0;
unsigned long lastVtbiUpdateTime = 0;

// [FIX, 2026-09-30] deliveredVolumeML itself was already being tracked
// correctly every loop() (below) - the auto-stop-at-target logic was
// never actually broken. The "VTBI: X mL left" LCD line, however, was
// only ever drawn ONCE, at the moment dose entry finished - nothing
// redrew it while the pump was actually running, so it displayed the
// starting value, frozen, for the entire infusion. Refreshed on its
// own short interval (not every loop() iteration) to avoid needless
// I2C traffic, same convention as BATTERY_REPORT_INTERVAL_MS.
unsigned long lastVtbiDisplayUpdate = 0;
#define VTBI_DISPLAY_UPDATE_INTERVAL_MS 1000

// -----------------------------------------------------
// CALIBRATION JOG - "CAL:<steps>" over Serial. Runs the motor forward
// for EXACTLY the requested number of step pulses at a fixed slow
// speed, then stops, so you can physically measure how far the
// plunger actually moved and check it against STEPS_PER_ML. Counts
// real pulses issued by stepper.runSpeed() rather than timing a rate,
// so the step count is exact regardless of loop jitter.
// -----------------------------------------------------
bool calibrationActive = false;
long calibrationStepsTarget = 0;
long calibrationStepsDone = 0;

// -----------------------------------------------------
// RETRACT (syringe reload) - "RETRACT" over Serial, or keypad 'C' from
// idle. Runs the motor in REVERSE (toward HOME) until HOME_LIMIT_PIN
// triggers, E-STOP fires, or RETRACT_MAX_STEPS is hit as a backup
// ceiling - whichever comes first. Unlike every other motion in this
// firmware, this is the one place a negative stepper speed is
// deliberately used; applyRate()/SET_RATE never support reverse.
// -----------------------------------------------------
bool retractActive = false;
long retractStepsDone = 0;

// -----------------------------------------------------
// SETTINGS MENU (keypad 'B' from idle) - mute toggle + a shortcut to
// RETRACT, so retract doesn't have to be memorized as a raw key. Only
// entered from idle (same pumpShouldRun guard as '0'/mode-change), and
// drawn once on entry like every other screen in this firmware (see
// showIdleScreen()'s call sites) rather than redrawn every loop().
// -----------------------------------------------------
bool settingsMenuActive = false;

Preferences prefs;
bool soundEnabled = false;  // Buzzer/sound in OFF position for now

// [ADDED 2026-10-10] Replaces the earlier WIFI_ENABLED compile-time
// flag used to diagnose the battery-only reboot-loop issue (confirmed
// cause: WiFi's TX current spikes browning out the battery/regulator
// path - see the WIRELESS CONTROL defines block). A persisted runtime
// setting, toggled from the Settings menu ('B' -> '2') exactly like
// soundEnabled above, is more useful than a compile flag for this:
// flip it off before an unpowered-by-USB/battery-only run until the
// hardware fix (bulk capacitor at the ESP32's own 3.3V rail) is in
// place, flip it back on once it is - no re-flash needed either way.
// Defaults to true (matches "works fine on USB/laptop power" - the
// current, immediate need) and takes effect on the NEXT boot, same as
// WiFi init always has (setupWifi() only ever runs once, from setup()).
bool wifiEnabled = true;   // loaded from NVS in setup(), see prefs.begin() there

// -----------------------------------------------------
// BATTERY SCREEN (keypad 'V'/'*' from idle OR while actively running -
// checking battery mid-infusion is legitimate and intentionally not
// blocked). [FIX, 2026-09-30] showBatteryOnLCD() used to just draw over
// the screen with no tracked state and no way back - if pressed while
// a dose was actively running, the operator was left staring at the
// battery screen with no hint of how to return, which is exactly what
// led to pressing 'A' out of desperation and landing in the dose-entry
// menu with the pump still running (see startDoseEntry()'s new guard
// above). Fixed the forgiving way: ANY key while this is true just
// dismisses it and restores the real current status - it does not
// also perform that key's normal action, so a confused keypress can
// never accidentally start/stop/change anything.
// -----------------------------------------------------
bool batteryScreenActive = false;



struct BatteryPoint {
  float voltage;
  float percent;
};

// [CALIBRATED 2026-11-11, updated same day] Re-mapped again to a wider
// pair of real bench reference points: 11.8V -> 100%, 8.5V -> 0%.
// Purely linear between the two; add intermediate points here later if
// the discharge curve turns out non-linear in practice and a smoother
// gauge is wanted. (batteryPercentFromVoltage()'s floor-point fix
// still applies and is harmless here since the floor point's own
// percent is 0.0 again, same as before that fix was needed.)
BatteryPoint batteryCurve[] = {
  {11.8, 100.0},
  {8.5, 0.0}
};

const int batteryCurvePoints = sizeof(batteryCurve) / sizeof(batteryCurve[0]);


// ============================================================
// LCD HELPERS
// ============================================================

void lcdClearLine(byte row) {
  lcd.setCursor(0, row);
  lcd.print("                ");
}

// This is a 1604 (16 columns) display. Never print past column 16 -
// on real HD44780-style 16x4 addressing, text that runs past the
// visible width doesn't just get cut off harmlessly, it keeps writing
// into DDRAM that can map onto a DIFFERENT physical row, corrupting
// whatever was there. Clamping here once, in the shared helpers,
// means every call site (including anything added later) is safe by
// construction instead of relying on manually counting characters.
#define LCD_COLS 16

void lcdLine(byte row, const char* text) {
  lcdClearLine(row);
  lcd.setCursor(0, row);
  String s = String(text);
  if (s.length() > LCD_COLS) {
    s = s.substring(0, LCD_COLS);
  }
  lcd.print(s);
}

void lcdLineString(byte row, String text) {
  lcdClearLine(row);
  lcd.setCursor(0, row);
  if (text.length() > LCD_COLS) {
    text = text.substring(0, LCD_COLS);
  }
  lcd.print(text);
}


// ============================================================
// BUZZER
// ============================================================

// Non-blocking buzzer - beep() used to block via delay(), which stacked
// up whenever several beeps fired close together (whether from the
// repeated-keypad-firing bug or just legitimate quick successive state
// changes). beep() now only starts a sequence; updateBuzzer() (called
// every loop() iteration, same pattern as updateLeds()'s red flash)
// actually toggles the pin over time without blocking anything else.
// Calling beep() again while one is still running restarts it fresh
// (the newest requested beep wins) - a deliberate simplification, not
// a queue.
bool buzzerActive = false;
bool buzzerCurrentlyOn = false;
byte buzzerBeepsRemaining = 0;
unsigned long buzzerBeepDurationMs = 0;
unsigned long buzzerLastToggle = 0;

// [UPDATED - Settings menu, 'B'->'1'] alwaysAudible=false (the default)
// means this beep is routine keypad/UI feedback and is silenced by the
// mute setting. alwaysAudible=true means this beep reports a genuine
// safety event or system state change (E-STOP, watchdog, end limit,
// VTBI complete, boot-ready) and always sounds regardless of the mute
// setting - explicitly opted into per call site below, not the default.
void beep(byte times, unsigned long duration, bool alwaysAudible = false) {

  if (!soundEnabled && !alwaysAudible) {
    return;
  }

  buzzerBeepsRemaining = times;
  buzzerBeepDurationMs = duration;

  digitalWrite(BUZZER_PIN, HIGH);
  buzzerCurrentlyOn = true;
  buzzerLastToggle = millis();
  buzzerActive = true;
}

void updateBuzzer() {

  if (!buzzerActive) {
    return;
  }

  if (millis() - buzzerLastToggle < buzzerBeepDurationMs) {
    return;   // not time to toggle yet
  }

  if (buzzerCurrentlyOn) {

    digitalWrite(BUZZER_PIN, LOW);
    buzzerCurrentlyOn = false;
    buzzerLastToggle = millis();
    buzzerBeepsRemaining--;

    if (buzzerBeepsRemaining == 0) {
      buzzerActive = false;
    }
  }
  else {

    digitalWrite(BUZZER_PIN, HIGH);
    buzzerCurrentlyOn = true;
    buzzerLastToggle = millis();
  }
}


// ============================================================
// BATTERY
// ============================================================

float readBatteryVoltage() {

  // analogReadMilliVolts(), not plain analogRead() - see connection_test.ino
  // for why: the ESP32 ADC's own nonlinearity under 11dB attenuation caused
  // a measured ~12% under-read (12.03V real vs 10.63V computed) using the
  // naive count->voltage math. analogReadMilliVolts() applies the chip's
  // factory calibration curve instead.
  const int numSamples = 16;
  long totalMillivolts = 0;

  for (int i = 0; i < numSamples; i++) {
    totalMillivolts += analogReadMilliVolts(BATTERY_ADC_PIN);
    delay(2);
  }

  float avgMillivolts = (float)totalMillivolts / numSamples;
  float adcVoltage = avgMillivolts / 1000.0;

  return adcVoltage * (DIVIDER_R1_OHMS + DIVIDER_R2_OHMS) / DIVIDER_R2_OHMS
         * BATTERY_VOLTAGE_CALIBRATION_FACTOR;
}

void updateBatteryReading() {

  float raw = readBatteryVoltage();

  if (smoothedBatteryVoltage < 0) {
    smoothedBatteryVoltage = raw;
  }
  else {
    smoothedBatteryVoltage =
      BATTERY_EMA_ALPHA * raw + (1.0 - BATTERY_EMA_ALPHA) * smoothedBatteryVoltage;
  }
}

float batteryPercentFromVoltage(float voltage) {

  if (voltage >= batteryCurve[0].voltage) {
    return 100.0;
  }

  // [FIX 2026-11-11] Was hardcoded "return 0.0" here - fine while the
  // last curve point's own percent WAS 0.0 (the old table), but wrong
  // in general: this branch fires for anything AT OR BELOW the last
  // point's voltage, which previously silently discarded that point's
  // real percent value. Now that the last point is {10.0V, 5%} rather
  // than {10.5V, 0%}, a reading of exactly 10.0V must still show 5%,
  // not 0% - only voltages genuinely below the last charted point
  // should floor out, and they floor at THAT point's percent, not an
  // assumed zero.
  if (voltage <= batteryCurve[batteryCurvePoints - 1].voltage) {
    return batteryCurve[batteryCurvePoints - 1].percent;
  }

  for (int i = 0; i < batteryCurvePoints - 1; i++) {

    float vHigh = batteryCurve[i].voltage;
    float vLow = batteryCurve[i + 1].voltage;

    if (voltage <= vHigh && voltage >= vLow) {

      float pHigh = batteryCurve[i].percent;
      float pLow = batteryCurve[i + 1].percent;
      float fraction = (voltage - vLow) / (vHigh - vLow);

      return pLow + fraction * (pHigh - pLow);
    }
  }

  return 0.0;
}

void showBatteryOnLCD() {

  float percent = batteryPercentFromVoltage(smoothedBatteryVoltage);

  lcd.clear();
  lcdLineString(0, "BATTERY STATUS");
  lcdLineString(1, "Voltage: " + String(smoothedBatteryVoltage, 2) + "V");
  lcdLineString(2, "Level: " + String(percent, 0) + "%");

  int filledBlocks = (int)((percent / 100.0) * 16.0);
  String bar = "";

  for (int i = 0; i < 16; i++) {
    bar += (i < filledBlocks) ? (char)255 : '-';
  }

  lcdLineString(3, bar);
  beep(1, 50);
}


// ============================================================
// STATUS LEDS
// ============================================================

bool homeLimitActive() {
  return digitalRead(HOME_LIMIT_PIN) == LOW;
}

bool endLimitActive() {
  return digitalRead(END_LIMIT_PIN) == LOW;
}

void updateLeds() {

  digitalWrite(GREEN_LED_PIN, emergencyStopLatched ? LOW : HIGH);

  bool redShouldFlash =
    emergencyStopLatched || homeLimitActive() || endLimitActive();

  if (redShouldFlash) {

    unsigned long now = millis();

    if (now - lastRedToggle >= RED_FLASH_INTERVAL_MS) {
      redLedState = !redLedState;
      digitalWrite(RED_LED_PIN, redLedState ? HIGH : LOW);
      lastRedToggle = now;
    }
  }
  else {
    digitalWrite(RED_LED_PIN, LOW);
    redLedState = false;
  }
}


// ============================================================
// RATE <-> STEPPER SPEED CONVERSION
// ============================================================

float rateToStepsPerSecond(float rateMLHr) {

  float mlPerSecond = rateMLHr / 3600.0;
  return mlPerSecond * STEPS_PER_ML;
}


// ============================================================
// WIRELESS CONTROL - globals, reply() helper, connection management
// ============================================================
// [ADDED 2026-10-07] Single authoritative wireless control client -
// a second incoming connection is refused while one is active, so the
// pump never has two different senders able to issue conflicting
// commands at once (a documented real-world hazard class for
// wireless-enabled infusion pumps, not a hypothetical). Wired Serial
// and this one wireless client both exist side by side, but both feed
// the SAME handleSerialLine() parser and the SAME applyRate()/
// watchdog machinery - there is still only ever one place a rate
// decision is actually applied, regardless of which transport it
// arrived on.
WiFiServer controlServer(TCP_CONTROL_PORT);
WiFiClient activeControlClient;
String wifiLineBuffer = "";
unsigned long lastWifiReconnectCheck = 0;
bool wifiEverConnected = false;

// reply() is used ONLY at the handful of command-response/safety-
// event sites a remote client actually needs to see (SET_RATE/
// SET_WEIGHT_DOSE/SET_VTBI_PROGRAM/STOP/GET_STATUS replies,
// applyRate()'s own warnings, E-STOP/watchdog/limit notices) - it is
// NOT a wholesale replacement of every Serial.print() in this file.
// The many purely-local LCD/keypad-menu diagnostic lines elsewhere
// stay Serial-only, unchanged, exactly as before this feature.
void reply(const String &msg) {

  Serial.println(msg);

  if (activeControlClient && activeControlClient.connected()) {
    activeControlClient.println(msg);
  }
}

void setupWifi() {

  Serial.print("WiFi: connecting to ");
  Serial.print(WIFI_SSID);
  Serial.println(" ...");

  WiFi.mode(WIFI_STA);

  // [ADDED 2026-10-10] See WIFI_TX_POWER's own comment above - this is
  // the actual fix for the battery-only reboot-loop issue (reported
  // after WiFi was added, only on battery, never on USB - a textbook
  // brownout-from-current-spike signature). Set BEFORE begin() so the
  // very first connection attempt (which also transmits) is already
  // capped, not just steady-state operation afterward.
  WiFi.setTxPower(WIFI_TX_POWER);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startAttempt = millis();

  while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < WIFI_CONNECT_TIMEOUT_MS) {
    delay(250);
  }

  if (WiFi.status() == WL_CONNECTED) {

    wifiEverConnected = true;
    controlServer.begin();

    Serial.print("WiFi connected. IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("TCP control port: ");
    Serial.println(TCP_CONTROL_PORT);
  }
  else {
    // Wired/keypad operation must never depend on WiFi succeeding -
    // boot continues normally either way. maintainWifi() (called every
    // loop()) will keep retrying in the background.
    Serial.println("WiFi connect timed out - continuing without wireless control.");
    Serial.println("Wired Serial and the keypad are unaffected.");
  }

  lastWifiReconnectCheck = millis();
}

// Called every loop() iteration, regardless of operatingMode, so a
// wireless client can connect (and handleModeSelection() can poll it)
// even before a mode is chosen - same reasoning Serial already gets
// polled during mode selection. Bounded by
// WIFI_RECONNECT_CHECK_INTERVAL_MS so a dropped link never costs more
// than one cheap status check per loop() iteration between retries.
void maintainWifi() {

  if (millis() - lastWifiReconnectCheck >= WIFI_RECONNECT_CHECK_INTERVAL_MS) {

    lastWifiReconnectCheck = millis();

    if (WiFi.status() != WL_CONNECTED) {

      // Non-blocking reattempt - unlike setupWifi()'s one-time boot
      // wait (acceptable there), nothing here may stall loop(), since
      // stepper.runSpeed() needs calling very frequently for smooth
      // motion.
      WiFi.reconnect();
      return;
    }

    if (!wifiEverConnected) {
      wifiEverConnected = true;
      controlServer.begin();
      Serial.print("WiFi connected. IP address: ");
      Serial.println(WiFi.localIP());
    }
  }

  if (!wifiEverConnected || WiFi.status() != WL_CONNECTED) {
    return;
  }

  // Accept a new connection only if no control client is currently
  // active - a second simultaneous sender is refused outright, not
  // queued or merged (see the "single authoritative wireless control
  // client" comment above).
  if (controlServer.hasClient()) {

    WiFiClient incoming = controlServer.available();

    if (activeControlClient && activeControlClient.connected()) {
      incoming.println("REJECTED: another wireless control client is already connected.");
      incoming.stop();
    }
    else {
      activeControlClient = incoming;
      wifiLineBuffer = "";
      Serial.print("Wireless control client connected: ");
      Serial.println(activeControlClient.remoteIP());
    }
  }

  if (activeControlClient && !activeControlClient.connected()) {
    Serial.println("Wireless control client disconnected.");
    activeControlClient.stop();
    wifiLineBuffer = "";
  }
}

// Mirrors loop()'s own Serial accumulate-until-newline block exactly
// (same pattern, same E-STOP guard), dispatched through the same
// handleSerialLine() - see loop() call site.
void pollWifiControlClient() {

  if (!(activeControlClient && activeControlClient.connected())) {
    return;
  }

  while (activeControlClient.available() > 0) {

    char c = activeControlClient.read();

    if (c == '\n') {

      if (!emergencyStopLatched || wifiLineBuffer == "R" || wifiLineBuffer == "r") {
        handleSerialLine(wifiLineBuffer);
      }
      else {
        reply("Command ignored - E-STOP active. Send R.");
      }

      wifiLineBuffer = "";
    }
    else if (c != '\r') {
      wifiLineBuffer += c;
    }
  }
}


// ============================================================
// DRIVER CONTROL
// ============================================================

void enableDriver() {

  digitalWrite(EN_PIN, DRIVER_ENABLED);
  driverEnabled = true;
}

void disableDriver() {

  digitalWrite(EN_PIN, DRIVER_DISABLED);
  driverEnabled = false;

  stepper.setSpeed(0);
}


// ============================================================
// APPLY A COMMANDED RATE
// ============================================================
// Central place all rate changes go through - clamps to the safe
// range, direction is always FORWARD for infusion (this firmware
// does not support commanding reverse via SET_RATE - reverse only
// happens implicitly if you wire it that way; add a DIRECTION
// field to the protocol later if withdrawal/priming needs it).

void applyRate(float requestedMLHr) {

  float clamped = requestedMLHr;

  if (clamped < MIN_FLOW_ML_HR) {
    clamped = MIN_FLOW_ML_HR;
  }

  if (clamped > MAX_FLOW_ML_HR) {

    reply("WARNING: requested rate " + String(requestedMLHr, 2)
      + " mL/hr exceeds MAX_FLOW_ML_HR - clamped to " + String(MAX_FLOW_ML_HR, 1));

    clamped = MAX_FLOW_ML_HR;
  }

  commandedRateMLHr = clamped;
  pumpShouldRun = (clamped > 0.0);

  if (emergencyStopLatched) {
    // E-STOP always wins - accept and store the rate, but do not
    // actually drive the motor until reset() is pressed.
    return;
  }

  if (pumpShouldRun && endLimitActive()) {

    reply("Rate REJECTED: END limit active.");
    pumpShouldRun = false;
    return;
  }

  if (pumpShouldRun) {

    enableDriver();
    stepper.setSpeed(rateToStepsPerSecond(commandedRateMLHr));
  }
  else {

    stepper.setSpeed(0);
  }
}


// ============================================================
// E-STOP
// ============================================================

void triggerEmergencyStop() {

  if (!emergencyStopLatched) {

    emergencyStopLatched = true;

    stepper.setSpeed(0);
    disableDriver();

    pumpShouldRun = false;
    vtbiActive = false;
    calibrationActive = false;
    retractActive = false;
    settingsMenuActive = false;
    batteryScreenActive = false;

    reply("");
    reply("!!! E-STOP ACTIVE - pump stopped !!!");
    reply("");

    lcd.clear();
    lcdLine(0, "!!! E-STOP !!!");
    lcdLine(1, "PUMP STOPPED");
    lcdLine(2, "Release E-STOP");
    lcdLine(3, "Press R to reset");

    beep(3, 100, true);   // E-STOP - always audible, not affected by the keypad-sound mute
  }
}

void checkEmergencyStop() {

  if (digitalRead(E_STOP_PIN) == LOW) {
    triggerEmergencyStop();
  }
}

void resetEmergencyStop() {

  if (digitalRead(E_STOP_PIN) == LOW) {

    reply("Cannot reset: physical E-STOP still pressed.");
    beep(2, 100, true);   // E-STOP subsystem - always audible
    return;
  }

  emergencyStopLatched = false;

  // Do NOT auto-resume the last commanded rate on reset - require a
  // fresh SET_RATE (or keypad entry) after E-STOP is cleared, so a
  // stale/old command can't silently restart the motor. Also drop out
  // of manual mode and cancel any dose entry in progress - a safety
  // reset should always require a deliberate, fresh decision from
  // whoever (PC or clinician) takes control again.
  commandedRateMLHr = 0.0;
  pumpShouldRun = false;
  manualModeActive = false;
  vtbiActive = false;
  calibrationActive = false;
  retractActive = false;
  entryMode = ENTRY_NONE;
  entryBuffer = "";
  disableDriver();

  // Force a fresh, deliberate mode choice too - same reasoning as not
  // auto-resuming a rate.
  operatingMode = MODE_UNSELECTED;

  reply("E-STOP reset. Please select a mode again.");

  beep(2, 80, true);   // E-STOP subsystem - always audible
  showModeSelectScreen();
}


// ============================================================
// LIMIT ENFORCEMENT (during a continuous run, checked every loop)
// ============================================================

void enforceLimits() {

  if (emergencyStopLatched) {
    return;
  }

  // Infusion is always FORWARD - so only the END limit (syringe fully
  // depressed) can stop a forward run here. HOME limit is handled
  // separately, in loop()'s own RETRACT block, since retract is the
  // only motion that ever travels toward HOME.
  //
  // [FIX, 2026-10-01] retractActive must be EXCLUDED from this check,
  // not just defensively cleared inside it. Retract reuses pumpShouldRun
  // (see startRetract()) so it IS "pumpShouldRun" too - without this
  // exclusion, starting a retract while the carriage was already
  // sitting AT the end limit (syringe fully depressed - exactly when
  // you'd most want to retract) got killed by this very check on the
  // very next enforceLimits() call, before a single reverse step ever
  // ran: pumpShouldRun had just been set true by startRetract(), and
  // endLimitActive() was still true because the carriage hadn't moved
  // away yet. Retract moving away from END is always safe - there is
  // no physical reason to stop it, so it's simply excluded here rather
  // than "allowed to clear itself after the fact."
  if (pumpShouldRun && !retractActive && endLimitActive()) {

    reply("END LIMIT reached - stopping pump.");

    stepper.setSpeed(0);
    disableDriver();

    pumpShouldRun = false;
    commandedRateMLHr = 0.0;
    calibrationActive = false;
    retractActive = false;

    lcdLine(3, "END LIMIT STOP");
    beep(2, 80, true);   // safety stop - always audible
  }
}


// ============================================================
// WATCHDOG
// ============================================================

void checkWatchdog() {

  if (!pumpShouldRun) {
    return;   // nothing running, nothing to watch
  }

  if (manualModeActive) {
    return;   // clinician set this rate directly - no PC in the loop to go quiet
  }

  if (calibrationActive) {
    return;   // one-off bench jog, not a PC-driven rate - never watchdog it
  }

  if (retractActive) {
    return;   // same reasoning - a reload jog, not a PC-driven rate
  }

  if (millis() - lastRateCommandTime > WATCHDOG_TIMEOUT_MS) {

    reply("WATCHDOG: no SET_RATE/SET_WEIGHT_DOSE/SET_VTBI_PROGRAM received in time - stopping pump.");

    stepper.setSpeed(0);
    disableDriver();

    pumpShouldRun = false;
    commandedRateMLHr = 0.0;

    lcdLine(3, "WATCHDOG STOP");
    beep(3, 60, true);   // safety stop - always audible

    vtbiActive = false;
  }
}


// ============================================================
// VTBI (Rate + VTBI program) - ported from the same accounting
// VirtualPatient.update_delivered_volume() already does on the Python
// side: accumulate delivered_volume = rate x elapsed_time, stop the
// instant the target is reached.
// ============================================================

void updateVtbiTracking() {

  if (!vtbiActive || !pumpShouldRun) {
    lastVtbiUpdateTime = millis();   // don't let a paused/stopped period count as elapsed time later
    return;
  }

  unsigned long now = millis();
  unsigned long elapsedMs = now - lastVtbiUpdateTime;
  lastVtbiUpdateTime = now;

  deliveredVolumeML += commandedRateMLHr * (elapsedMs / 3600000.0);

  if (deliveredVolumeML >= vtbiTargetML) {

    Serial.print("VTBI COMPLETE: delivered ");
    Serial.print(deliveredVolumeML, 2);
    Serial.print(" / ");
    Serial.print(vtbiTargetML, 2);
    Serial.println(" mL - stopping pump.");

    stepper.setSpeed(0);
    disableDriver();

    pumpShouldRun = false;
    commandedRateMLHr = 0.0;
    vtbiActive = false;

    lcd.clear();
    lcdLine(0, "VTBI COMPLETE");
    lcdLineString(1, String(deliveredVolumeML, 1) + " mL delivered");
    lcdLine(3, "0 = change mode");

    beep(3, 100, true);   // infusion-complete alert - always audible
    return;
  }

  // [FIX, 2026-09-30] Periodic live refresh - see lastVtbiDisplayUpdate's
  // declaration above for why this is needed. Deliberately just this
  // one row, not a full lcd.clear() + redraw of the whole screen, to
  // avoid visible flicker every second and to avoid touching rows 0/1/3
  // in case something else legitimately changed them since the run
  // started (e.g. driverEnabled toggling). Also unifies VOL+TIME mode's
  // row 2 (which used to statically echo the entered "Xml/Ymin" and
  // never update at all) onto the same live countdown Rate+VTBI mode
  // already showed at entry time - a real improvement there, not just
  // a fix, since the entered values were already confirmed once during
  // entry and a live remaining-volume readout is more useful while running.
  if (millis() - lastVtbiDisplayUpdate >= VTBI_DISPLAY_UPDATE_INTERVAL_MS) {
    lastVtbiDisplayUpdate = millis();
    lcdLineString(2, "VTBI:" + String(vtbiTargetML - deliveredVolumeML, 1) + "mL left");
  }
}


// ============================================================
// CALIBRATION JOG ("CAL:<steps>" over Serial)
// ============================================================
// Purpose: verify STEPS_PER_ML against reality. The design assumes
// 1/16 microstepping (3200 steps/turn) but MS1/MS2/MS3 aren't wired
// to the ESP32, so the A4988 module's actual microstepping setting is
// unverified. Command a known step count, physically measure how far
// the plunger moved (or how much volume came out), and compare
// against calibrationExpectedML() below - if they don't match,
// recalculate STEPS_PER_ML from the real result.

void startCalibration(long steps) {

  if (steps <= 0) {
    Serial.println("CAL REJECTED: steps must be a positive integer, e.g. CAL:3200");
    return;
  }

  if (emergencyStopLatched) {
    Serial.println("CAL REJECTED: E-STOP active.");
    return;
  }

  if (pumpShouldRun) {
    Serial.println("CAL REJECTED: pump is already running - stop it first (# or STOP).");
    return;
  }

  if (endLimitActive()) {
    Serial.println("CAL REJECTED: END limit active - cannot move forward.");
    return;
  }

  calibrationActive = true;
  calibrationStepsTarget = steps;
  calibrationStepsDone = 0;

  // Reuse the normal continuous-run pipeline (enableDriver + setSpeed +
  // the runSpeed() call at the bottom of loop()) - just gated by a step
  // count instead of a rate/VTBI condition. manualModeActive=true
  // exempts this from the watchdog, same reasoning as a keypad-set
  // rate: there's no PC in the loop that could "go quiet".
  pumpShouldRun = true;
  manualModeActive = true;
  commandedRateMLHr = CALIBRATION_SPEED_ML_HR;

  enableDriver();
  stepper.setSpeed(rateToStepsPerSecond(CALIBRATION_SPEED_ML_HR));

  Serial.print("CAL: running ");
  Serial.print(steps);
  Serial.println(" steps at calibration speed. Do not touch the keypad/PC until it reports done.");
  Serial.print("CAL: at the CURRENT STEPS_PER_ML=");
  Serial.print(STEPS_PER_ML, 2);
  Serial.print(", expected travel = ");
  Serial.print(steps / STEPS_PER_ML, 4);
  Serial.println(" mL - measure the REAL result once it stops and compare.");

  lcd.clear();
  lcdLine(0, "CALIBRATION JOG");
  lcdLineString(1, String(steps) + " steps");
  lcdLine(2, "Measure travel");
  lcdLine(3, "when it stops..");

  beep(1, 60);
}

void finishCalibration() {

  stepper.setSpeed(0);
  disableDriver();

  pumpShouldRun = false;
  manualModeActive = false;
  commandedRateMLHr = 0.0;
  calibrationActive = false;

  Serial.print("CAL DONE: commanded ");
  Serial.print(calibrationStepsTarget);
  Serial.println(" steps.");
  Serial.println("CAL: now physically measure the actual plunger travel (mm) or expelled volume (mL).");
  Serial.print("CAL: expected at current settings = ");
  Serial.print(calibrationStepsTarget / STEPS_PER_ML, 4);
  Serial.println(" mL.");
  Serial.println("CAL: if the real number differs, real_STEPS_PER_ML = steps / real_mL - update");
  Serial.println("     MICROSTEPS_PER_TURN (or STEPS_PER_ML directly) in the header to match.");

  lcd.clear();
  lcdLine(0, "CAL DONE");
  lcdLineString(1, String(calibrationStepsTarget) + " steps sent");
  lcdLine(2, "Measure now,");
  lcdLine(3, "check Serial");

  beep(2, 100);
}


// ============================================================
// RETRACT (syringe reload) - "RETRACT" over Serial, or keypad 'C'
// ============================================================
// Purpose: withdraw the plunger back toward HOME so a spent syringe
// can be swapped for a fresh one, without needing to power down or
// manually push the carriage back by hand. This is the "future
// priming/retract feature" the HOME limit switch was originally wired
// for but never used - see enforceLimits().
//
// Direction is REVERSE, which nothing else in this firmware ever
// commands - applyRate()/SET_RATE only ever drive forward (see that
// function's own header comment). A negative stepper speed is used
// here deliberately, for this one action only.

void startRetract() {

  if (emergencyStopLatched) {
    Serial.println("RETRACT REJECTED: E-STOP active.");
    return;
  }

  if (pumpShouldRun) {
    Serial.println("RETRACT REJECTED: pump is already running - stop it first (# or STOP).");
    return;
  }

  if (homeLimitActive()) {
    Serial.println("RETRACT REJECTED: HOME limit already active - nothing to retract.");
    return;
  }

  retractActive = true;
  retractStepsDone = 0;

  // Same "reuse the continuous-run pipeline, exempt from the watchdog"
  // approach as startCalibration() - this is a deliberate bench action
  // with no PC in the loop, not a dosing command.
  pumpShouldRun = true;
  manualModeActive = true;
  commandedRateMLHr = 0.0;   // this motion isn't a dose - keep the display honest

  enableDriver();
  stepper.setSpeed(-rateToStepsPerSecond(RETRACT_SPEED_ML_HR));   // NEGATIVE = reverse

  Serial.println("RETRACT: reversing toward HOME. Will stop automatically at the HOME limit.");
  Serial.print("RETRACT: safety ceiling = ");
  Serial.print(RETRACT_MAX_STEPS);
  Serial.println(" steps, in case the HOME switch fails to trigger.");

  lcd.clear();
  lcdLine(0, "RETRACTING...");
  lcdLine(1, "Syringe reload");
  lcdLine(2, "Auto-stops at");
  lcdLine(3, "HOME limit");

  beep(1, 60);
}

void finishRetract(bool hitHomeLimit) {

  stepper.setSpeed(0);
  disableDriver();

  pumpShouldRun = false;
  manualModeActive = false;
  retractActive = false;

  Serial.print("RETRACT DONE: ");
  Serial.print(retractStepsDone);
  Serial.print(" steps. Stop reason: ");
  Serial.println(hitHomeLimit ? "HOME limit reached." : "safety step ceiling reached (HOME limit never triggered - check the switch/wiring).");

  lcd.clear();
  lcdLine(0, "RETRACT DONE");
  lcdLine(1, hitHomeLimit ? "At HOME limit" : "STEP LIMIT HIT");
  lcdLine(2, "Reload syringe,");
  lcdLine(3, "then press 0");

  beep(hitHomeLimit ? 2 : 4, hitHomeLimit ? 100 : 60);
}


// ============================================================
// STATUS
// ============================================================

// Takes a Print& rather than hardcoding Serial so the exact same
// status line can be written to the active wireless control client
// too - see printStatusLine() below, which is the function everything
// else in this file actually calls.
void printStatusLineTo(Print &out) {

  out.print("STATUS,mode=");
  out.print(operatingMode == MODE_MANUAL ? "MANUAL"
    : operatingMode == MODE_AUTONOMOUS ? "AUTONOMOUS" : "UNSELECTED");
  out.print(",rate=");
  out.print(commandedRateMLHr, 2);
  out.print(",running=");
  out.print(pumpShouldRun ? 1 : 0);
  out.print(",estop=");
  out.print(emergencyStopLatched ? 1 : 0);
  out.print(",home=");
  out.print(homeLimitActive() ? 1 : 0);
  out.print(",end=");
  out.print(endLimitActive() ? 1 : 0);
  out.print(",battery_v=");
  out.print(smoothedBatteryVoltage, 2);
  out.print(",battery_pct=");
  out.print(batteryPercentFromVoltage(smoothedBatteryVoltage), 0);
  out.print(",vtbi_active=");
  out.print(vtbiActive ? 1 : 0);
  out.print(",vtbi_target_ml=");
  out.print(vtbiTargetML, 2);
  out.print(",vtbi_delivered_ml=");
  out.print(deliveredVolumeML, 2);
  out.print(",cal_active=");
  out.print(calibrationActive ? 1 : 0);
  out.print(",cal_steps_done=");
  out.print(calibrationStepsDone);
  out.print(",cal_steps_target=");
  out.print(calibrationStepsTarget);
  out.print(",retract_active=");
  out.print(retractActive ? 1 : 0);
  out.print(",retract_steps_done=");
  out.print(retractStepsDone);
  out.print(",sound_enabled=");
  out.print(soundEnabled ? 1 : 0);
  out.print(",wifi_enabled=");
  out.print(wifiEnabled ? 1 : 0);
  out.print(",position=");
  out.println(stepper.currentPosition());
}

void printStatusLine() {

  printStatusLineTo(Serial);

  if (activeControlClient && activeControlClient.connected()) {
    printStatusLineTo(activeControlClient);
  }
}


// ============================================================
// SERIAL LINE PROTOCOL
// ============================================================

void handleSerialLine(String line) {

  line.trim();

  if (line.length() == 0) {
    return;
  }

  // Mode not yet chosen - reproduces exactly what the old, separate
  // handleModeSelection() parser used to do before both transports
  // were unified onto this one dispatcher (see handleModeSelection()):
  // an explicit MODE: line selects directly; any other recognized PC
  // command auto-selects Autonomous but is itself DROPPED this cycle,
  // not executed - the sender will simply send its next line moments
  // later, now that a mode is selected.
  if (operatingMode == MODE_UNSELECTED) {

    if (line == "MODE:MANUAL") {
      selectMode(MODE_MANUAL);
    }
    else if (line == "MODE:AUTO"
             || line.startsWith("SET_RATE:")
             || line.startsWith("SET_WEIGHT_DOSE:")
             || line.startsWith("SET_VTBI_PROGRAM:")
             || line == "GET_STATUS") {
      selectMode(MODE_AUTONOMOUS);
    }

    return;
  }

  if (line.startsWith("SET_RATE:")) {

    if (operatingMode == MODE_MANUAL) {

      reply("REJECTED: system is in MANUAL mode - PC control is disabled.");
      reply("Press 0 on the keypad (from idle) to switch to AUTONOMOUS mode.");
      return;
    }

    // Now known to be MODE_AUTONOMOUS - the PC always takes back control
    // the instant it sends a command; cancel any in-progress keypad
    // entry and drop out of manual mode, so the watchdog resumes
    // applying to this (now PC-driven) rate.
    if (entryMode != ENTRY_NONE) {
      cancelDoseEntry("PC sent SET_RATE");
    }
    manualModeActive = false;
    vtbiActive = false;   // PC control has no concept of the keypad's VTBI - abandon it, not pause it
    calibrationActive = false;   // a real SET_RATE always takes priority over a bench jog
    retractActive = false;       // same reasoning - a dosing command always wins over a reload jog

    float requested = line.substring(9).toFloat();

    lastRateCommandTime = millis();

    applyRate(requested);

    reply("OK,rate=" + String(commandedRateMLHr, 2));

    lcd.clear();
    lcdLine(0, "PUMP RUNNING");
    lcdLineString(1, "Rate: " + String(commandedRateMLHr, 2) + " mL/hr");
    lcdLine(2, driverEnabled ? "Driver: ENABLED" : "Driver: DISABLED");
    lcdLine(3, emergencyStopLatched ? "E-STOP ACTIVE" : "OK");

    return;
  }

  if (line.startsWith("SET_WEIGHT_DOSE:")) {

    if (operatingMode == MODE_MANUAL) {

      reply("REJECTED: system is in MANUAL mode - PC control is disabled.");
      reply("Press 0 on the keypad (from idle) to switch to AUTONOMOUS mode.");
      return;
    }

    // Format: SET_WEIGHT_DOSE:<weight_kg>,<dose_value>,<unit>, unit is
    // MLHR or UGKGMIN - the PC-facing equivalent of the keypad's
    // weight-dose program (ENTRY_WEIGHT -> ENTRY_UNIT_SELECT ->
    // ENTRY_RATE), sent as one already-validated atomic command rather
    // than digit-by-digit keypad entry. No multi-turn correction on a
    // bad value, unlike the keypad - the client must resend a
    // corrected full command; it owns its own input validation.
    String body = line.substring(16);   // length of "SET_WEIGHT_DOSE:"
    int firstComma = body.indexOf(',');
    int secondComma = body.indexOf(',', firstComma + 1);

    if (firstComma < 0 || secondComma < 0) {
      reply("REJECTED: malformed SET_WEIGHT_DOSE - expected <weight_kg>,<dose_value>,<unit>");
      return;
    }

    float weightKg = body.substring(0, firstComma).toFloat();
    float doseValue = body.substring(firstComma + 1, secondComma).toFloat();
    String unit = body.substring(secondComma + 1);
    unit.trim();

    if (weightKg <= 0.0 || weightKg > 300.0) {   // sanity bound, same as the keypad's own check
      reply("REJECTED: invalid weight (must be 0-300 kg).");
      return;
    }

    bool isUgKgMin;

    if (unit == "MLHR") {
      isUgKgMin = false;
    }
    else if (unit == "UGKGMIN") {
      isUgKgMin = true;
    }
    else {
      reply("REJECTED: unit must be MLHR or UGKGMIN.");
      return;
    }

    if (entryMode != ENTRY_NONE) {
      cancelDoseEntry("PC sent SET_WEIGHT_DOSE");
    }

    // Same conversion formula the keypad's weight-dose program uses.
    float rateMLHr = isUgKgMin
      ? (doseValue * weightKg * 60.0) / DRUG_CONCENTRATION_MCG_ML
      : doseValue;

    // Unlike the keypad path (which sets manualModeActive=true -
    // "clinician set this directly, no PC in the loop to go quiet"),
    // this arrived over a PC/remote channel, so it STAYS
    // watchdog-covered - a dropped wireless link is more likely than a
    // dropped USB cable, and the existing watchdog is exactly the
    // mechanism that makes that safe. Do not copy manualModeActive=true
    // here.
    // Standalone manual weight-dose program from desktop UI:
    // exempt from 3s PC watchdog so it runs to completion (cleared by STOP or SET_RATE).
    manualModeActive = true;
    vtbiActive = false;
    calibrationActive = false;
    retractActive = false;
    lastRateCommandTime = millis();

    applyRate(rateMLHr);

    reply("OK,weight_dose,rate=" + String(commandedRateMLHr, 2));

    lcd.clear();
    lcdLine(0, "PUMP RUNNING");
    lcdLineString(1, "Rate: " + String(commandedRateMLHr, 2) + " mL/hr");
    lcdLineString(2, "Weight: " + String(weightKg, 1) + " kg");
    lcdLine(3, emergencyStopLatched ? "E-STOP ACTIVE" : "OK");

    return;
  }

  if (line.startsWith("SET_VTBI_PROGRAM:")) {

    if (operatingMode == MODE_MANUAL) {

      reply("REJECTED: system is in MANUAL mode - PC control is disabled.");
      reply("Press 0 on the keypad (from idle) to switch to AUTONOMOUS mode.");
      return;
    }

    // Format: SET_VTBI_PROGRAM:<rate_ml_hr>,<vtbi_ml> - one atomic
    // command, the PC-facing equivalent of the keypad's Rate+VTBI
    // program (ENTRY_RATE -> ENTRY_VTBI), sent fully-formed instead of
    // the keypad's two-step rate-then-VTBI entry.
    String body = line.substring(17);   // length of "SET_VTBI_PROGRAM:"
    int comma = body.indexOf(',');

    if (comma < 0) {
      reply("REJECTED: malformed SET_VTBI_PROGRAM - expected <rate_ml_hr>,<vtbi_ml>");
      return;
    }

    float rateMLHr = body.substring(0, comma).toFloat();
    float vtbiMl = body.substring(comma + 1).toFloat();

    if (vtbiMl <= 0.0) {
      reply("REJECTED: invalid VTBI (must be > 0 mL).");
      return;
    }

    if (entryMode != ENTRY_NONE) {
      cancelDoseEntry("PC sent SET_VTBI_PROGRAM");
    }

    vtbiActive = true;
    vtbiTargetML = vtbiMl;
    deliveredVolumeML = 0.0;
    lastVtbiUpdateTime = millis();
    lastVtbiDisplayUpdate = millis();

    // Standalone manual Rate+VTBI program from desktop UI:
    // exempt from 3s PC watchdog so VTBI runs to completion (cleared by STOP or SET_RATE).
    manualModeActive = true;
    calibrationActive = false;
    retractActive = false;
    lastRateCommandTime = millis();

    applyRate(rateMLHr);

    reply("OK,vtbi,rate=" + String(commandedRateMLHr, 2) + ",vtbi=" + String(vtbiTargetML, 2));

    lcd.clear();
    lcdLine(0, "PUMP RUNNING");
    lcdLineString(1, "Rate: " + String(commandedRateMLHr, 2) + " mL/hr");
    lcdLineString(2, "VTBI: " + String(vtbiTargetML, 1) + " mL");
    lcdLine(3, emergencyStopLatched ? "E-STOP ACTIVE" : "OK");

    return;
  }

  if (line == "STOP") {

    stepper.setSpeed(0);
    disableDriver();

    pumpShouldRun = false;
    commandedRateMLHr = 0.0;
    vtbiActive = false;
    calibrationActive = false;
    retractActive = false;

    reply("OK,stopped");

    lcdLine(3, "STOPPED (cmd)");

    return;
  }

  if (line == "GET_STATUS") {
    printStatusLine();
    return;
  }

  if (line.startsWith("CAL:")) {
    long steps = line.substring(4).toInt();
    startCalibration(steps);
    return;
  }

  if (line == "RETRACT") {
    startRetract();
    return;
  }

  // Single-character legacy/manual commands, same as the keypad below.
  if (line.length() == 1) {
    handleKeypadOrManualCommand(line.charAt(0));
    return;
  }

  reply("Unrecognized command: " + line);
}


// ============================================================
// MANUAL / KEYPAD COMMANDS (available even during automated run)
// ============================================================

// -----------------------------------------------------
// MODE SELECTION (at boot, or re-entered from idle via '0')
// -----------------------------------------------------

void showModeSelectScreen() {

  lcd.clear();
  lcdLine(0, "SELECT MODE:");
  lcdLine(1, "A=Manual(keypad)");
  lcdLine(2, "B=Autonomous(PC)");
  lcdLine(3, "");

  Serial.println("SELECT MODE: send 'A'/'MODE:MANUAL' or 'B'/'MODE:AUTO',");
  Serial.println("or just send any SET_RATE/GET_STATUS to auto-select Autonomous.");
}

void selectMode(OperatingMode mode) {

  operatingMode = mode;

  Serial.println(mode == MODE_MANUAL
    ? "MODE: MANUAL selected."
    : "MODE: AUTONOMOUS selected. Awaiting SET_RATE from PC.");

  beep(1, 80);
  showIdleScreen();
}

void handleModeSelection() {

  char key = getNewKeyPress();

  if (key != NO_KEY) {
    Serial.print("KEY DETECTED: '");
    Serial.print(key);
    Serial.println("'");
    lcdLineString(3, "Key: " + String(key));
  }

  if (key == 'A' || key == 'a') {
    selectMode(MODE_MANUAL);
    return;
  }

  if (key == 'B' || key == 'b') {
    selectMode(MODE_AUTONOMOUS);
    return;
  }

  // esp32_bridge.py/native_app.py (wired) and any wireless control
  // client both have no keypad - let any of their commands during mode
  // selection auto-select Autonomous, or accept an explicit MODE: line.
  // Both transports feed the SAME handleSerialLine() (which recognizes
  // MODE:MANUAL/MODE:AUTO itself, gated on operatingMode==
  // MODE_UNSELECTED - see there) rather than this function parsing
  // mode selection a second, separate way.
  while (Serial.available() > 0) {

    char c = Serial.read();

    if (c == '\n') {
      String line = serialLineBuffer;
      serialLineBuffer = "";
      handleSerialLine(line);
    }
    else if (c != '\r') {
      serialLineBuffer += c;
    }
  }

  if (wifiEnabled && activeControlClient && activeControlClient.connected()) {

    while (activeControlClient.available() > 0) {

      char c = activeControlClient.read();

      if (c == '\n') {
        String line = wifiLineBuffer;
        wifiLineBuffer = "";
        handleSerialLine(line);
      }
      else if (c != '\r') {
        wifiLineBuffer += c;
      }
    }
  }
}


void showIdleScreen() {

  // Timing instrumentation from the earlier intermittent-freeze
  // investigation removed - it hasn't recurred since, and the evidence
  // pointed to something environmental (USB/Serial connection state),
  // not this function.

  lcd.clear();

  if (emergencyStopLatched) {
    lcdLine(0, "E-STOP ACTIVE");
    lcdLine(3, "Press R to reset");
    return;
  }

  lcdLine(0, operatingMode == MODE_MANUAL ? "MODE: MANUAL" : "MODE: AUTONOMOUS");

  if (vtbiActive) {
    lcdLineString(1, "Rate: " + String(commandedRateMLHr, 2) + " mL/hr");
    lcdLineString(2, "VTBI:" + String(vtbiTargetML - deliveredVolumeML, 1) + "mL left");
  }
  else {
    lcdLineString(1, "Rate: " + String(commandedRateMLHr, 2) + " mL/hr");
    lcdLine(2, driverEnabled ? "Driver: ENABLED" : "Driver: DISABLED");
  }

  lcdLine(3, operatingMode == MODE_MANUAL ? "A=dose 0=mode" : "0 = change mode");
}


// -----------------------------------------------------
// SETTINGS MENU - keypad 'B' from idle (see handleKeypadOrManualCommand).
// Drawn once on entry, same pattern as every other screen here - loop()
// does not redraw this every iteration, so it stays put until a key
// changes it or E-STOP overwrites it (triggerEmergencyStop() clears
// settingsMenuActive so a stale flag can't misroute the next keypress).
// -----------------------------------------------------
void showSettingsMenu() {

  // [RESHUFFLED 2026-10-10] Added a WiFi ON/OFF toggle ('2') - pushed
  // Retract to '3' to make room. 16x4 LCD has no spare line, so row 3
  // now carries both the retract shortcut AND the back option in one
  // line ("3=Retract D=back" is exactly 16 characters).
  lcd.clear();
  lcdLine(0, "SETTINGS");
  lcdLine(1, soundEnabled ? "1=Sound: ON" : "1=Sound: OFF");
  lcdLine(2, wifiEnabled ? "2=WiFi: ON" : "2=WiFi: OFF");
  lcdLine(3, "3=Retract D=back");
}

void handleSettingsMenuKey(char cmd) {

  switch (cmd) {

    case '1':
      soundEnabled = !soundEnabled;
      prefs.putBool("sound", soundEnabled);
      // If sound was just turned OFF, this call is itself silenced by
      // the new setting (beep() checks soundEnabled first) - the
      // screen redraw below is the confirmation in that case instead.
      beep(1, 60);
      showSettingsMenu();
      break;

    case '2':
      // [ADDED 2026-10-10] Persisted WiFi on/off - see wifiEnabled's
      // own comment (confirmed root cause of a battery-only reboot
      // loop: WiFi's TX current spikes). Takes effect on the NEXT
      // boot, not immediately - setupWifi()/WiFi.begin() only ever run
      // once, from setup(). No blocking delay()/confirmation screen
      // here deliberately - this project has a documented history of
      // blocking calls inside keypad handlers causing real freezes
      // (see getNewKeyPress()'s own comment) and would also stop
      // checkEmergencyStop() from being serviced for that long. The
      // "takes effect next boot" note goes to Serial instead; the
      // redrawn menu's own ON/OFF label is the on-screen confirmation.
      wifiEnabled = !wifiEnabled;
      prefs.putBool("wifi", wifiEnabled);
      Serial.print("WiFi setting changed to: ");
      Serial.print(wifiEnabled ? "ON" : "OFF");
      Serial.println(" - takes effect on next power cycle/reset.");
      beep(1, 60);
      showSettingsMenu();
      break;

    case '3':
      // Same action as the standalone 'C' shortcut - startRetract()
      // has its own idle/E-STOP/already-running guards and draws its
      // own screen, so just hand off to it exactly like 'C' does.
      settingsMenuActive = false;
      startRetract();
      break;

    case 'D':
    case 'd':
    case '0':
      settingsMenuActive = false;
      showIdleScreen();
      break;

    default:
      break;
  }
}


// -----------------------------------------------------
// MANUAL DOSE ENTRY - Option B state machine
//   A            -> start entry (from idle only)
//   0-9          -> digit
//   *            -> decimal point (while entering a number)
//   #            -> confirm current field, advance to the next
//   B / C        -> select unit (mL/hr / mcg/kg/min), at the unit step
//   D            -> cancel, at any point, back to idle
// -----------------------------------------------------

void startDoseEntry() {

  if (emergencyStopLatched) {
    Serial.println("Cannot start dose entry - E-STOP active.");
    beep(2, 70);
    return;
  }

  // [FIX, 2026-09-30] This guard was missing entirely - every other
  // "start a new action" key ('0' mode-change, 'B' Settings, 'C'
  // Retract) already refuses while pumpShouldRun is true, but this one
  // didn't, so pressing 'A' during an ACTIVE infusion (e.g. while
  // trying to get back from the battery screen, see showBatteryOnLCD)
  // dropped straight into the SELECT PROGRAM menu with the motor still
  // physically running, undisplayed, in the background. Worse than the
  // confusing screen alone: completing that entry would have reset
  // deliveredVolumeML to 0 and applied a brand new rate on top of a
  // dose that was still running - a real dosing-accounting hazard, not
  // just a UI issue.
  if (pumpShouldRun) {
    Serial.println("Cannot start dose entry - the pump is already running. Stop it first (# or STOP).");
    beep(2, 70);
    return;
  }

  if (calibrationActive) {
    Serial.println("Cannot start dose entry - a calibration jog is running. Wait for it to finish.");
    beep(2, 70);
    return;
  }

  if (retractActive) {
    Serial.println("Cannot start dose entry - a retract jog is running. Wait for it to finish.");
    beep(2, 70);
    return;
  }

  if (operatingMode != MODE_MANUAL) {
    Serial.println("Cannot start dose entry - system is in AUTONOMOUS mode. Press 0 to change mode.");
    beep(2, 70);
    return;
  }

  entryMode = ENTRY_PROGRAM_SELECT;
  entryBuffer = "";
  selectedProgram = PROGRAM_NONE;

  lcd.clear();
  lcdLine(0, "SELECT PROGRAM:");
  lcdLine(1, "1=Rate+VTBI");
  lcdLine(2, "2=Wt 3=Vol+Time");
  lcdLine(3, "D = cancel");
}

void cancelDoseEntry(const char* reason) {

  entryMode = ENTRY_NONE;
  entryBuffer = "";

  Serial.print("Dose entry cancelled: ");
  Serial.println(reason);

  beep(1, 100);
  showIdleScreen();
}

// Backspace for numeric entry - deletes the last typed character (digit
// or decimal point) and refreshes the live display, so a mistyped digit
// doesn't force cancelling the whole entry with D and starting over from
// the program-select screen. Bound to 'C' in every plain numeric-entry
// state (weight, rate, VTBI, volume, time) - free everywhere it's used
// this way, since 'C' only means something else (select ug/kg/min) in
// the separate ENTRY_UNIT_SELECT state, which has no buffer to edit.
void backspaceEntryBuffer(byte displayRow) {

  if (entryBuffer.length() > 0) {
    entryBuffer.remove(entryBuffer.length() - 1);
  }

  lcdLineString(displayRow, entryBuffer);
}

void handleEntryKey(char key) {

  if (key == 'D' || key == 'd') {
    cancelDoseEntry("user pressed D");
    return;
  }

  // -------------------- PROGRAM SELECT --------------------
  if (entryMode == ENTRY_PROGRAM_SELECT) {

    if (key == '1') {
      selectedProgram = PROGRAM_RATE_VTBI;
      entryMode = ENTRY_RATE;
      entryBuffer = "";
      lcd.clear();
      lcdLine(0, "RATE + VTBI");
      lcdLine(1, "Rate (mL/hr):");
      lcdLine(3, "*=. #nxt D=can");
      return;
    }

    if (key == '2') {
      selectedProgram = PROGRAM_WEIGHT_DOSE;
      entryMode = ENTRY_WEIGHT;
      entryBuffer = "";
      lcd.clear();
      lcdLine(0, "WEIGHT DOSE");
      lcdLine(1, "Weight (kg):");
      lcdLine(3, "*=. #nxt D=can");
      return;
    }

    if (key == '3') {
      selectedProgram = PROGRAM_VOLUME_TIME;
      entryMode = ENTRY_VT_VOLUME;
      entryBuffer = "";
      lcd.clear();
      lcdLine(0, "VOLUME+TIME");
      lcdLine(1, "Volume (mL):");
      lcdLine(3, "*=. #nxt D=can");
      return;
    }

    return;   // ignore anything else while waiting for 1/2/3
  }

  // -------------------- WEIGHT --------------------
  if (entryMode == ENTRY_WEIGHT) {

    if (key == '#') {

      float weight = entryBuffer.toFloat();

      if (weight <= 0.0 || weight > 300.0) {   // sanity bound, not a clinical limit

        Serial.println("Invalid weight - re-enter.");
        lcdLine(2, "Invalid! Retry:");
        entryBuffer = "";
        beep(2, 70);
        return;
      }

      enteredWeightKg = weight;
      entryBuffer = "";
      entryMode = ENTRY_UNIT_SELECT;

      lcd.clear();
      lcdLineString(0, "Weight: " + String(enteredWeightKg, 1) + " kg");
      lcdLine(1, "B=mL/hr");
      lcdLine(2, "C=ug/kg/min");
      lcdLine(3, "D=cancel");
      return;
    }

    if (key == 'C' || key == 'c') {
      backspaceEntryBuffer(2);
      return;
    }

    if ((key >= '0' && key <= '9') || key == '*') {
      entryBuffer += (key == '*') ? '.' : key;
      lcdLineString(2, entryBuffer);
    }

    return;
  }

  // -------------------- UNIT SELECT --------------------
  if (entryMode == ENTRY_UNIT_SELECT) {

    if (key == 'B' || key == 'b') {
      enteredUnitIsMcgKgMin = false;
      entryMode = ENTRY_RATE;
      entryBuffer = "";
      lcd.clear();
      lcdLine(0, "Rate (mL/hr):");
      lcdLine(3, "*=. #OK D=can");
      return;
    }

    if (key == 'C' || key == 'c') {
      enteredUnitIsMcgKgMin = true;
      entryMode = ENTRY_RATE;
      entryBuffer = "";
      lcd.clear();
      lcdLine(0, "Rate(ug/kg/min):");
      lcdLine(3, "*=. #OK D=can");
      return;
    }

    return;   // ignore anything else while waiting for B/C
  }

  // -------------------- RATE --------------------
  // Shared by both programs. Rate+VTBI: confirming here moves on to the
  // VTBI step, nothing is applied to the motor yet. Weight-dose:
  // confirming here converts (if needed) and applies immediately, same
  // behavior as before this feature was added.
  if (entryMode == ENTRY_RATE) {

    if (key == '#') {

      float enteredValue = entryBuffer.toFloat();

      if (enteredValue < 0.0) {

        Serial.println("Invalid rate - re-enter.");
        lcdLine(1, "Invalid! Retry:");
        entryBuffer = "";
        beep(2, 70);
        return;
      }

      if (selectedProgram == PROGRAM_RATE_VTBI) {

        pendingRateMLHr = enteredValue;
        entryBuffer = "";
        entryMode = ENTRY_VTBI;

        lcd.clear();
        lcdLineString(0, "Rate: " + String(pendingRateMLHr, 2) + " mL/hr");
        lcdLine(1, "VTBI (mL):");
        lcdLine(3, "*=. #OK D=can");
        return;
      }

      // PROGRAM_WEIGHT_DOSE - unchanged from before this feature.
      float rateMLHr = enteredUnitIsMcgKgMin
        ? (enteredValue * enteredWeightKg * 60.0) / DRUG_CONCENTRATION_MCG_ML
        : enteredValue;

      entryMode = ENTRY_NONE;
      entryBuffer = "";
      manualModeActive = true;

      applyRate(rateMLHr);

      Serial.print("MANUAL dose set: weight=");
      Serial.print(enteredWeightKg, 1);
      Serial.print("kg, entered=");
      Serial.print(enteredValue, 3);
      Serial.println(enteredUnitIsMcgKgMin ? " ug/kg/min" : " mL/hr");
      Serial.print("-> commanded rate = ");
      Serial.print(commandedRateMLHr, 2);
      Serial.println(" mL/hr");

      lcd.clear();
      lcdLine(0, "MANUAL MODE");
      lcdLineString(1, "Rate: " + String(commandedRateMLHr, 2) + " mL/hr");
      lcdLineString(2, "Weight: " + String(enteredWeightKg, 1) + " kg");
      lcdLine(3, driverEnabled ? "Driver: ENABLED" : "Driver: DISABLED");

      beep(2, 80);
      return;
    }

    if (key == 'C' || key == 'c') {
      backspaceEntryBuffer(1);
      return;
    }

    if ((key >= '0' && key <= '9') || key == '*') {
      entryBuffer += (key == '*') ? '.' : key;
      lcdLineString(1, entryBuffer);
    }

    return;
  }

  // -------------------- VTBI --------------------
  // Rate+VTBI program only. Confirming here is the point the rate
  // actually gets applied to the motor - the same "apply everything at
  // once, on the final confirm" pattern the weight-dose program uses.
  if (entryMode == ENTRY_VTBI) {

    if (key == '#') {

      float enteredVtbi = entryBuffer.toFloat();

      if (enteredVtbi <= 0.0) {

        Serial.println("Invalid VTBI - re-enter.");
        lcdLine(2, "Invalid! Retry:");
        entryBuffer = "";
        beep(2, 70);
        return;
      }

      entryMode = ENTRY_NONE;
      entryBuffer = "";
      manualModeActive = true;

      vtbiActive = true;
      vtbiTargetML = enteredVtbi;
      deliveredVolumeML = 0.0;
      lastVtbiUpdateTime = millis();
      lastVtbiDisplayUpdate = millis();

      applyRate(pendingRateMLHr);

      Serial.print("RATE+VTBI set: rate=");
      Serial.print(commandedRateMLHr, 2);
      Serial.print(" mL/hr, VTBI=");
      Serial.print(vtbiTargetML, 2);
      Serial.println(" mL");

      lcd.clear();
      lcdLine(0, "RATE+VTBI MODE");
      lcdLineString(1, "Rate: " + String(commandedRateMLHr, 2) + " mL/hr");
      lcdLineString(2, "VTBI:" + String(vtbiTargetML, 1) + "mL left");
      lcdLine(3, driverEnabled ? "Driver: ENABLED" : "Driver: DISABLED");

      beep(2, 80);
      return;
    }

    if (key == 'C' || key == 'c') {
      backspaceEntryBuffer(2);
      return;
    }

    if ((key >= '0' && key <= '9') || key == '*') {
      entryBuffer += (key == '*') ? '.' : key;
      lcdLineString(2, entryBuffer);
    }

    return;
  }

  // -------------------- VOLUME+TIME: VOLUME --------------------
  // Volume-Time program only. Volume is entered first; confirming here
  // moves on to the time step, same "collect everything, apply on the
  // final confirm" pattern as Rate+VTBI.
  if (entryMode == ENTRY_VT_VOLUME) {

    if (key == '#') {

      float enteredVolume = entryBuffer.toFloat();

      if (enteredVolume <= 0.0) {

        Serial.println("Invalid volume - re-enter.");
        lcdLine(2, "Invalid! Retry:");
        entryBuffer = "";
        beep(2, 70);
        return;
      }

      pendingVolumeML = enteredVolume;
      entryBuffer = "";
      entryMode = ENTRY_VT_TIME;

      lcd.clear();
      lcdLineString(0, "Volume: " + String(pendingVolumeML, 2) + " mL");
      lcdLine(1, "Time (min):");
      lcdLine(3, "*=. #OK D=can");
      return;
    }

    if (key == 'C' || key == 'c') {
      backspaceEntryBuffer(2);
      return;
    }

    if ((key >= '0' && key <= '9') || key == '*') {
      entryBuffer += (key == '*') ? '.' : key;
      lcdLineString(2, entryBuffer);
    }

    return;
  }

  // -------------------- VOLUME+TIME: TIME --------------------
  // Volume-Time program only. Confirming here is the point the pump
  // actually computes rate = volume/time and applies it - same
  // apply-on-final-confirm pattern as the other programs. Reuses the
  // exact same VTBI auto-stop tracking as Rate+VTBI, since "run at this
  // rate until this volume is delivered" is exactly what this program
  // reduces to once the rate is computed.
  if (entryMode == ENTRY_VT_TIME) {

    if (key == '#') {

      float enteredMinutes = entryBuffer.toFloat();

      if (enteredMinutes <= 0.0) {

        Serial.println("Invalid time - re-enter.");
        lcdLine(2, "Invalid! Retry:");
        entryBuffer = "";
        beep(2, 70);
        return;
      }

      float computedRateMLHr = pendingVolumeML / (enteredMinutes / 60.0);

      entryMode = ENTRY_NONE;
      entryBuffer = "";
      manualModeActive = true;

      vtbiActive = true;
      vtbiTargetML = pendingVolumeML;
      deliveredVolumeML = 0.0;
      lastVtbiUpdateTime = millis();
      lastVtbiDisplayUpdate = millis();

      applyRate(computedRateMLHr);   // clamps to MAX_FLOW_ML_HR and warns if the
                                      // requested volume/time combo isn't achievable

      Serial.print("VOLUME+TIME set: volume=");
      Serial.print(pendingVolumeML, 2);
      Serial.print(" mL over ");
      Serial.print(enteredMinutes, 1);
      Serial.print(" min -> rate=");
      Serial.print(commandedRateMLHr, 2);
      Serial.println(" mL/hr");

      lcd.clear();
      lcdLine(0, "VOL+TIME MODE");
      lcdLineString(1, "Rate: " + String(commandedRateMLHr, 2) + " mL/hr");
      lcdLineString(2, String(pendingVolumeML, 1) + "mL/" + String(enteredMinutes, 0) + "min");
      lcdLine(3, driverEnabled ? "Driver: ENABLED" : "Driver: DISABLED");

      beep(2, 80);
      return;
    }

    if (key == 'C' || key == 'c') {
      backspaceEntryBuffer(2);
      return;
    }

    if ((key >= '0' && key <= '9') || key == '*') {
      entryBuffer += (key == '*') ? '.' : key;
      lcdLineString(2, entryBuffer);
    }

    return;
  }
}


void handleKeypadOrManualCommand(char cmd) {

  switch (cmd) {

    case 'A':
    case 'a':
      startDoseEntry();
      break;

    case 'R':
    case 'r':
      resetEmergencyStop();
      break;

    case 'T':
    case 't':
      printStatusLine();
      break;

    case 'V':
    case 'v':
    case '*':
      batteryScreenActive = true;
      showBatteryOnLCD();
      break;

    case '#':
      // Idle soft-stop - same effect as sending "STOP" over Serial, but
      // reachable from the keypad without touching the PC or the
      // hardware E-STOP.
      stepper.setSpeed(0);
      disableDriver();
      pumpShouldRun = false;
      commandedRateMLHr = 0.0;
      manualModeActive = false;
      vtbiActive = false;
      calibrationActive = false;
      retractActive = false;
      Serial.println("Pump stopped (keypad #).");
      showIdleScreen();
      beep(1, 80);
      break;

    case 'C':
    case 'c':
      // Syringe reload - reverse jog toward HOME. Idle-only, same
      // reasoning as CAL: (a deliberate bench action, not a dosing
      // command) - startRetract() itself rejects if the pump is
      // already running.
      startRetract();
      break;

    case 'B':
    case 'b':
      // Settings menu - idle-only, same reasoning as '0' below (don't
      // let menu browsing coexist with whichever control source
      // currently has the pump moving).
      if (pumpShouldRun) {
        Serial.println("Cannot open Settings while the pump is running - stop it first (# or STOP).");
        beep(2, 70);
        break;
      }
      settingsMenuActive = true;
      showSettingsMenu();
      break;

    case '0':
      // Return to mode selection - only while nothing is actively
      // running, so switching modes can't be used to sidestep whichever
      // control source currently has the pump moving.
      if (pumpShouldRun) {
        Serial.println("Cannot change mode while the pump is running - stop it first (# or STOP).");
        beep(2, 70);
        break;
      }
      operatingMode = MODE_UNSELECTED;
      Serial.println("Returning to mode selection.");
      showModeSelectScreen();
      break;

    default:
      Serial.print("Key/char received (no automated action bound): ");
      Serial.println(cmd);
      break;
  }
}


// ============================================================
// I2C SCANNER (boot-time only, same as connection_test.ino)
// ============================================================

void scanI2C() {

  Serial.println("Scanning I2C bus...");
  byte count = 0;

  for (byte address = 1; address < 127; address++) {

    Wire.beginTransmission(address);

    if (Wire.endTransmission() == 0) {
      Serial.print("  Found I2C device at 0x");
      Serial.println(address, HEX);
      count++;
    }
  }

  if (count == 0) {
    Serial.println("  No I2C devices found - check LCD wiring/power.");
  }
}


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  // [FIX, 2026-10-01] Was Serial.setTxTimeoutMs(0) - removed, does not
  // compile against Serial: HardwareSerial has no setTxTimeoutMs()
  // member (that method is specific to the native USB-CDC Serial
  // class, which this board doesn't use - see the big header comment,
  // "Serial -> Serial"). It's also not needed here: the blocking-
  // write-when-nobody's-listening problem it worked around is a CDC
  // software-buffering quirk, not something real UART hardware has -
  // Serial just shifts bits out the TX pin regardless of whether
  // anything is reading them.

  delay(500);

  Serial.println();
  Serial.println("==========================================");
  Serial.println(" SMART SYRINGE INFUSION PUMP - REAL FIRMWARE V1");
  Serial.println(" ESP32-S3");
  Serial.println("==========================================");
  Serial.println();

  Serial.print("Derived STEPS_PER_ML = ");
  Serial.println(STEPS_PER_ML, 2);
  Serial.println("(verify against your board's actual microstepping setting - see header comment)");
  Serial.println();

  // Settings menu ('B' from idle) - false = read/write mode (not
  // read-only), "pump" is just a namespace name for this firmware's
  // NVS entries, doesn't need to match anything else.
  prefs.begin("pump", false);

  // [2026-11-11] Deliberately NOT loaded from NVS - sound always
  // starts OFF at boot by design, regardless of whatever was toggled
  // on in a previous session. Toggling it on via the Settings menu
  // ('B' -> '1') still works for the rest of THIS session; it's just
  // never remembered into the next boot. (prefs.putBool("sound", ...)
  // in handleSettingsMenuKey() still writes a value here, which is
  // harmless - it's simply never read back.)
  soundEnabled = false;
  Serial.println("Sound: OFF at boot (by design - toggle on via Settings menu if needed).");

  wifiEnabled = prefs.getBool("wifi", true);
  Serial.print("WiFi enabled (from saved settings): ");
  Serial.println(wifiEnabled ? "YES" : "NO");

  // -------------------- SAFETY INPUTS --------------------
  pinMode(E_STOP_PIN, INPUT_PULLUP);
  pinMode(HOME_LIMIT_PIN, INPUT_PULLUP);
  pinMode(END_LIMIT_PIN, INPUT_PULLUP);

  // -------------------- BUZZER --------------------
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // -------------------- STATUS LEDS --------------------
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(RED_LED_PIN, LOW);

  // -------------------- BATTERY ADC --------------------
  analogReadResolution(12);
  analogSetPinAttenuation(BATTERY_ADC_PIN, ADC_11db);
  updateBatteryReading();

  // -------------------- A4988 --------------------
  pinMode(EN_PIN, OUTPUT);
  digitalWrite(EN_PIN, DRIVER_DISABLED);
  driverEnabled = false;

  // -------------------- I2C + LCD --------------------
  Wire.begin(SDA_PIN, SCL_PIN);
  scanI2C();

  lcd.begin(16, 4);
  lcd.setBacklight(255);
  lcd.clear();
  lcdLine(0, "SMART PUMP");
  lcdLine(1, "Real Firmware V1");
  lcdLine(2, "Initializing...");

  // -------------------- STEPPER --------------------
  // TESTED both directionInvert=true AND false on the bench - shaft
  // spun ANTICLOCKWISE (viewed from the front, wrong way, wanted
  // clockwise/push) either way. This proved the DIR pin isn't the
  // lever controlling rotation here (loose/wrong wire, or a driver
  // board that ignores it) - do NOT re-toggle this again expecting a
  // different result. Direction was actually fixed by swapping the
  // two leads of ONE motor coil pair at the driver connector instead.
  stepper.setPinsInverted(false, false, false);

  // [UPDATED 2026-09-30] Was rateToStepsPerSecond(MAX_FLOW_ML_HR)*1.1
  // only - RETRACT_SPEED_ML_HR (250.0) is now ABOVE MAX_FLOW_ML_HR
  // (190.0), so the ceiling has to be based on whichever of the two is
  // larger, or AccelStepper::setSpeed()'s own internal constrain() to
  // _maxSpeed would silently cap the retract jog back down to the old
  // ceiling instead of actually reaching 250. Dosing itself is NOT
  // affected - applyRate() independently clamps every requested rate
  // to MAX_FLOW_ML_HR before it ever reaches setSpeed(), regardless of
  // what this ceiling is set to.
  stepper.setMaxSpeed(rateToStepsPerSecond(max(MAX_FLOW_ML_HR, RETRACT_SPEED_ML_HR)) * 1.1);   // small headroom
  stepper.setAcceleration(MOTOR_ACCELERATION_STEPS_S2);
  stepper.setSpeed(0);

  // -------------------- INITIAL E-STOP CHECK --------------------
  if (digitalRead(E_STOP_PIN) == LOW) {

    emergencyStopLatched = true;

    lcd.clear();
    lcdLine(0, "E-STOP PRESSED");
    lcdLine(1, "Pump disabled");
    lcdLine(2, "Release E-STOP");
    lcdLine(3, "Press R to reset");
  }
  else {

    beep(2, 100, true);   // boot-ready chime - not a keypad interaction, always audible
    showModeSelectScreen();
  }

  lastRateCommandTime = millis();   // don't let the watchdog fire before any command was ever expected

  // -------------------- WIFI --------------------
  // After everything else is up (LCD/keypad/stepper/safety inputs) so
  // a slow/failed WiFi connect attempt never delays those being ready.
  // See setupWifi()'s own comment - boot continues regardless of
  // outcome. Gated by the persisted wifiEnabled setting (Settings menu
  // 'B' -> '2', see its own comment) - when off, the radio never
  // initializes at all, zero WiFi current draw. Loaded from NVS just
  // above, so this reflects whatever was last set via the keypad.
  if (wifiEnabled) {
    setupWifi();
  }
  else {
    Serial.println("WiFi disabled (Settings menu) - skipping radio init.");
  }

  Serial.println("Setup complete.");
  Serial.println();
}


// ============================================================
// MAIN LOOP
// ============================================================

void loop() {

  // 1. E-STOP - always checked first, wins over everything, even during
  // mode selection.
  checkEmergencyStop();

  // Buzzer must be serviced unconditionally, every loop iteration,
  // regardless of what state we're in below - beep() only starts a
  // sequence, this is what actually finishes it. If this were placed
  // after the mode-selection early-return, a beep triggered on the
  // mode-select screen would latch the buzzer HIGH forever, since
  // nothing would ever run the rest of its toggle sequence.
  updateBuzzer();

  // 1a. WiFi connection/reconnect housekeeping and accepting a new
  // control client - runs regardless of mode, so a client can connect
  // even before a mode is chosen (handleModeSelection() below polls it
  // too, same as it already polls Serial). Gated by the persisted
  // wifiEnabled setting - if it was off at boot, setupWifi() was never
  // called either, so there's nothing to maintain.
  if (wifiEnabled) {
    maintainWifi();
  }

  // 1b. Mode not yet chosen - show the menu and wait, skip everything
  // else (nothing can be commanded yet anyway - driver stays disabled).
  if (operatingMode == MODE_UNSELECTED) {
    handleModeSelection();
    return;
  }

  // 2. Limit enforcement during an active run
  enforceLimits();

  // 3. Watchdog - auto-stop if the PC has gone quiet
  checkWatchdog();

  // 3b. VTBI - auto-stop when the target volume has been delivered
  updateVtbiTracking();

  // 4. Status LEDs
  updateLeds();

  // 5. Battery, periodic
  if (millis() - lastBatteryReport >= BATTERY_REPORT_INTERVAL_MS) {
    updateBatteryReading();
    lastBatteryReport = millis();
  }

  // 6. Keypad (manual override commands only - R, T, V/*)
  char key = getNewKeyPress();

  if (key) {
    Serial.print("KEY DETECTED: '");
    Serial.print(key);
    Serial.println("'");

    if (batteryScreenActive) {

      // ANY key dismisses the battery screen and restores whatever the
      // real current status actually is (showIdleScreen() already
      // correctly reflects running-with-VTBI vs. truly idle vs.
      // driver-enabled state) - it does NOT also perform that key's
      // normal action, so a confused keypress here can never
      // accidentally start dose entry, open Settings, or stop the
      // pump. E-STOP already redraws its own screen and clears this
      // flag the instant it latches (triggerEmergencyStop()), so
      // reaching here with it still true can't actually happen -
      // guarded anyway, same defence-in-depth reason as the branches
      // below.
      batteryScreenActive = false;
      if (!emergencyStopLatched) {
        showIdleScreen();
      }
    }
    else if (settingsMenuActive) {

      // Same E-STOP guard shape as the dose-entry branch below -
      // triggerEmergencyStop() already clears settingsMenuActive the
      // instant it latches, so reaching here with it still true means
      // E-STOP was already active when the menu was entered (can't
      // actually happen, since 'B' is rejected during E-STOP the same
      // way '0' is) - handled anyway for the same defence-in-depth
      // reason dose-entry guards against it explicitly.
      if (emergencyStopLatched) {
        settingsMenuActive = false;
      }
      else {
        handleSettingsMenuKey(key);
      }
    }
    else if (entryMode != ENTRY_NONE) {

      // Mid dose-entry: E-STOP still cancels everything via
      // checkEmergencyStop() above, which does not touch entryMode
      // directly - guard here too so a stale entry can't complete
      // into applyRate() after E-STOP just latched this same loop.
      if (emergencyStopLatched) {
        cancelDoseEntry("E-STOP activated");
      }
      else {
        handleEntryKey(key);
      }
    }
    else if (!emergencyStopLatched || key == 'R' || key == 'r') {
      handleKeypadOrManualCommand(key);
    }
    else {
      Serial.println("Key ignored - E-STOP active. Press R.");
    }
  }

  // 7. Serial line protocol - accumulate until newline
  while (Serial.available() > 0) {

    char c = Serial.read();

    if (c == '\n') {

      if (!emergencyStopLatched || serialLineBuffer == "R" || serialLineBuffer == "r") {
        handleSerialLine(serialLineBuffer);
      }
      else {
        reply("Command ignored - E-STOP active. Send R.");
      }

      serialLineBuffer = "";
    }
    else if (c != '\r') {

      serialLineBuffer += c;
    }
  }

  // 7b. Wireless control client - same accumulate-until-newline
  // pattern as the Serial block above, dispatched through the same
  // handleSerialLine(). Gated by the persisted wifiEnabled setting.
  if (wifiEnabled) {
    pollWifiControlClient();
  }

  // 8. Run the stepper at whatever speed was last set (constant-speed
  // mode - NOT stepper.run(), which is for move()-to-target jogging).
  if (calibrationActive) {

    // Count real pulses issued (runSpeed() returns true exactly when it
    // steps) rather than timing a rate, so this is exact regardless of
    // loop jitter.
    if (stepper.runSpeed()) {

      calibrationStepsDone++;

      if (calibrationStepsDone >= calibrationStepsTarget) {
        finishCalibration();
      }
    }
  }
  else if (retractActive) {

    // Check the limit BEFORE stepping, not after - stops immediately
    // on the tick HOME becomes active rather than taking one extra
    // step past it.
    if (homeLimitActive()) {
      finishRetract(true);
    }
    else if (stepper.runSpeed()) {

      retractStepsDone++;

      if (retractStepsDone >= RETRACT_MAX_STEPS) {
        finishRetract(false);   // HOME never triggered - safety ceiling hit instead
      }
    }
  }
  else {
    stepper.runSpeed();
  }
}
