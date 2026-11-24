#include <Keypad.h>
#include <Wire.h>
#include <LiquidCrystal_PCF8574.h>
#include <AccelStepper.h>

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

  SERIAL PROTOCOL (from the PC, one line at a time):
    SET_RATE:<mL/hr>   e.g. "SET_RATE:12.5"
                       Clamped to [0, MAX_FLOW_ML_HR]. Resets
                       the watchdog timer.
    STOP               Immediately zero the rate and disable
                       the driver.
    GET_STATUS         Prints one status line (see printStatusLine()).
    CAL:<steps>        Calibration jog - runs exactly <steps> step
                       pulses forward at a fixed slow speed, then
                       stops. Only when idle (not mid-run), not
                       watchdog-guarded. See CALIBRATION note below.

  WATCHDOG:
    If no SET_RATE command arrives within WATCHDOG_TIMEOUT_MS,
    the pump automatically stops - protects against a PC crash,
    USB disconnect, or a hung control loop upstream.

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
        during entry: 0-9 = digit, * = decimal point, # = confirm/next,
                      D = cancel entire entry, C = backspace (delete
                      last digit typed - fixes a mistake without
                      cancelling and starting over),
                      B/C = select mL/hr or ug/kg/min (weight-dose unit
                      step only - C means "backspace" everywhere else,
                      since that step has no digits to edit)
    R = reset E-STOP (after physically releasing it)
    T = print status
    V / * = show battery on LCD (idle only - * means decimal point
            while a dose entry is in progress)
    # = soft-stop (idle only - same as sending STOP over Serial)

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
  ============================================================
*/


// ============================================================
// PIN DEFINITIONS
// ============================================================

// ---------------- KEYPAD ----------------

// Physical wiring:
// Row 1 -> GPIO 4
// Row 2 -> GPIO 5
// Row 3 -> GPIO 6
// Row 4 -> GPIO 7
// Col 1 -> GPIO 15
// Col 2 -> GPIO 16
// Col 3 -> GPIO 17
// Col 4 -> GPIO 18

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

#define BATTERY_ADC_PIN 3

#define DIVIDER_R1_OHMS 100000.0
#define DIVIDER_R2_OHMS 27000.0

#define ADC_VREF 3.3
#define ADC_MAX_COUNT 4095.0

#define BATTERY_REPORT_INTERVAL_MS 5000
#define BATTERY_EMA_ALPHA 0.2


// ============================================================
// PUMP / MOTOR SETTINGS
// ============================================================

#define DRIVER_ENABLED  LOW
#define DRIVER_DISABLED HIGH

#define MAX_FLOW_ML_HR 190.0
#define MIN_FLOW_ML_HR 0.0

#define SYRINGE_BORE_DIAMETER_MM 19.13
#define LEAD_SCREW_PITCH_MM_PER_TURN 8.0
#define MICROSTEPS_PER_TURN 3200.0

const float SYRINGE_BORE_AREA_MM2 =
  3.14159265 * (SYRINGE_BORE_DIAMETER_MM / 2.0) *
  (SYRINGE_BORE_DIAMETER_MM / 2.0);

const float ML_PER_TURN =
  (SYRINGE_BORE_AREA_MM2 * LEAD_SCREW_PITCH_MM_PER_TURN) / 1000.0;

const float STEPS_PER_ML = MICROSTEPS_PER_TURN / ML_PER_TURN;

#define MOTOR_ACCELERATION_STEPS_S2 200.0

#define WATCHDOG_TIMEOUT_MS 3000

#define DRUG_CONCENTRATION_MCG_ML 16.0

#define CALIBRATION_SPEED_ML_HR 150.0


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

byte rowPins[ROWS] = { ROW1, ROW2, ROW3, ROW4 };
byte colPins[COLS] = { COL1, COL2, COL3, COL4 };

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
// OPERATING MODE
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

char lastKeyPressed = NO_KEY;

char getNewKeyPress() {

  char key = keypad.getKey();

  if (key == NO_KEY) {
    lastKeyPressed = NO_KEY;
    return NO_KEY;
  }

  if (key == lastKeyPressed) {
    return NO_KEY;
  }

  lastKeyPressed = key;
  return key;
}


// -----------------------------------------------------
// MANUAL DOSE ENTRY
// -----------------------------------------------------

enum ManualProgram {
  PROGRAM_NONE,
  PROGRAM_RATE_VTBI,
  PROGRAM_WEIGHT_DOSE,
  PROGRAM_VOLUME_TIME
};

ManualProgram selectedProgram = PROGRAM_NONE;

enum EntryMode {
  ENTRY_NONE,
  ENTRY_PROGRAM_SELECT,
  ENTRY_WEIGHT,
  ENTRY_UNIT_SELECT,
  ENTRY_RATE,
  ENTRY_VTBI,
  ENTRY_VT_VOLUME,
  ENTRY_VT_TIME
};

EntryMode entryMode = ENTRY_NONE;
String entryBuffer = "";
float enteredWeightKg = 0.0;
bool enteredUnitIsMcgKgMin = false;
float pendingRateMLHr = 0.0;
float pendingVolumeML = 0.0;

bool manualModeActive = false;


// -----------------------------------------------------
// VTBI
// -----------------------------------------------------

bool vtbiActive = false;
float vtbiTargetML = 0.0;
float deliveredVolumeML = 0.0;
unsigned long lastVtbiUpdateTime = 0;


// -----------------------------------------------------
// CALIBRATION JOG
// -----------------------------------------------------

bool calibrationActive = false;
long calibrationStepsTarget = 0;
long calibrationStepsDone = 0;


// ============================================================
// BATTERY CALIBRATION TABLE
// ============================================================

struct BatteryPoint {
  float voltage;
  float percent;
};

BatteryPoint batteryCurve[] = {
  {12.6, 100.0},
  {12.3, 90.0},
  {12.0, 70.0},
  {11.7, 50.0},
  {11.4, 30.0},
  {11.1, 15.0},
  {10.5, 0.0}
};

const int batteryCurvePoints =
  sizeof(batteryCurve) / sizeof(batteryCurve[0]);


// ============================================================
// LCD HELPERS
// ============================================================

void lcdClearLine(byte row) {
  lcd.setCursor(0, row);
  lcd.print("                ");
}

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

bool buzzerActive = false;
bool buzzerCurrentlyOn = false;
byte buzzerBeepsRemaining = 0;
unsigned long buzzerBeepDurationMs = 0;
unsigned long buzzerLastToggle = 0;

void beep(byte times, unsigned long duration) {

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
    return;
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

  const int numSamples = 16;
  long totalMillivolts = 0;

  for (int i = 0; i < numSamples; i++) {
    totalMillivolts += analogReadMilliVolts(BATTERY_ADC_PIN);
    delay(2);
  }

  float avgMillivolts =
    (float)totalMillivolts / numSamples;

  float adcVoltage =
    avgMillivolts / 1000.0;

  return adcVoltage *
         (DIVIDER_R1_OHMS + DIVIDER_R2_OHMS) /
         DIVIDER_R2_OHMS;
}

void updateBatteryReading() {

  float raw = readBatteryVoltage();

  if (smoothedBatteryVoltage < 0) {
    smoothedBatteryVoltage = raw;
  }
  else {
    smoothedBatteryVoltage =
      BATTERY_EMA_ALPHA * raw +
      (1.0 - BATTERY_EMA_ALPHA) *
      smoothedBatteryVoltage;
  }
}

float batteryPercentFromVoltage(float voltage) {

  if (voltage >= batteryCurve[0].voltage) {
    return 100.0;
  }

  if (voltage <= batteryCurve[batteryCurvePoints - 1].voltage) {
    return 0.0;
  }

  for (int i = 0; i < batteryCurvePoints - 1; i++) {

    float vHigh = batteryCurve[i].voltage;
    float vLow = batteryCurve[i + 1].voltage;

    if (voltage <= vHigh && voltage >= vLow) {

      float pHigh = batteryCurve[i].percent;
      float pLow = batteryCurve[i + 1].percent;

      float fraction =
        (voltage - vLow) / (vHigh - vLow);

      return pLow + fraction * (pHigh - pLow);
    }
  }

  return 0.0;
}

void showBatteryOnLCD() {

  float percent =
    batteryPercentFromVoltage(smoothedBatteryVoltage);

  lcd.clear();

  lcdLineString(0, "BATTERY STATUS");
  lcdLineString(1,
    "Voltage: " +
    String(smoothedBatteryVoltage, 2) +
    "V");

  lcdLineString(2,
    "Level: " +
    String(percent, 0) +
    "%");

  int filledBlocks =
    (int)((percent / 100.0) * 16.0);

  String bar = "";

  for (int i = 0; i < 16; i++) {
    bar +=
      (i < filledBlocks) ?
      (char)255 : '-';
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

  digitalWrite(
    GREEN_LED_PIN,
    emergencyStopLatched ? LOW : HIGH
  );

  bool redShouldFlash =
    emergencyStopLatched ||
    homeLimitActive() ||
    endLimitActive();

  if (redShouldFlash) {

    unsigned long now = millis();

    if (now - lastRedToggle >= RED_FLASH_INTERVAL_MS) {

      redLedState = !redLedState;

      digitalWrite(
        RED_LED_PIN,
        redLedState ? HIGH : LOW
      );

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

  float mlPerSecond =
    rateMLHr / 3600.0;

  return mlPerSecond * STEPS_PER_ML;
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

void applyRate(float requestedMLHr) {

  float clamped = requestedMLHr;

  if (clamped < MIN_FLOW_ML_HR) {
    clamped = MIN_FLOW_ML_HR;
  }

  if (clamped > MAX_FLOW_ML_HR) {

    Serial.print("WARNING: requested rate ");
    Serial.print(requestedMLHr, 2);
    Serial.print(
      " mL/hr exceeds MAX_FLOW_ML_HR - clamped to "
    );
    Serial.println(MAX_FLOW_ML_HR, 1);

    clamped = MAX_FLOW_ML_HR;
  }

  commandedRateMLHr = clamped;
  pumpShouldRun = (clamped > 0.0);

  if (emergencyStopLatched) {
    return;
  }

  if (pumpShouldRun && endLimitActive()) {

    Serial.println(
      "Rate REJECTED: END limit active."
    );

    pumpShouldRun = false;
    return;
  }

  if (pumpShouldRun) {

    enableDriver();

    stepper.setSpeed(
      rateToStepsPerSecond(commandedRateMLHr)
    );
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

    Serial.println();
    Serial.println(
      "!!! E-STOP ACTIVE - pump stopped !!!"
    );
    Serial.println();

    lcd.clear();

    lcdLine(0, "!!! E-STOP !!!");
    lcdLine(1, "PUMP STOPPED");
    lcdLine(2, "Release E-STOP");
    lcdLine(3, "Press R to reset");

    beep(3, 100);
  }
}

void checkEmergencyStop() {

  if (digitalRead(E_STOP_PIN) == LOW) {
    triggerEmergencyStop();
  }
}

void resetEmergencyStop() {

  if (digitalRead(E_STOP_PIN) == LOW) {

    Serial.println(
      "Cannot reset: physical E-STOP still pressed."
    );

    beep(2, 100);
    return;
  }

  emergencyStopLatched = false;

  commandedRateMLHr = 0.0;
  pumpShouldRun = false;
  manualModeActive = false;
  vtbiActive = false;
  calibrationActive = false;
  entryMode = ENTRY_NONE;
  entryBuffer = "";

  disableDriver();

  operatingMode = MODE_UNSELECTED;

  Serial.println(
    "E-STOP reset. Please select a mode again."
  );

  beep(2, 80);
  showModeSelectScreen();
}


// ============================================================
// LIMIT ENFORCEMENT
// ============================================================

void enforceLimits() {

  if (emergencyStopLatched) {
    return;
  }

  if (pumpShouldRun && endLimitActive()) {

    Serial.println(
      "END LIMIT reached - stopping pump."
    );

    stepper.setSpeed(0);
    disableDriver();

    pumpShouldRun = false;
    commandedRateMLHr = 0.0;
    calibrationActive = false;

    lcdLine(3, "END LIMIT STOP");

    beep(2, 80);
  }
}


// ============================================================
// WATCHDOG
// ============================================================

void checkWatchdog() {

  if (!pumpShouldRun) {
    return;
  }

  if (manualModeActive) {
    return;
  }

  if (calibrationActive) {
    return;
  }

  if (millis() - lastRateCommandTime >
      WATCHDOG_TIMEOUT_MS) {

    Serial.println(
      "WATCHDOG: no SET_RATE received in time - stopping pump."
    );

    stepper.setSpeed(0);
    disableDriver();

    pumpShouldRun = false;
    commandedRateMLHr = 0.0;

    lcdLine(3, "WATCHDOG STOP");

    beep(3, 60);

    vtbiActive = false;
  }
}


// ============================================================
// VTBI
// ============================================================

void updateVtbiTracking() {

  if (!vtbiActive || !pumpShouldRun) {

    lastVtbiUpdateTime = millis();

    return;
  }

  unsigned long now = millis();

  unsigned long elapsedMs =
    now - lastVtbiUpdateTime;

  lastVtbiUpdateTime = now;

  deliveredVolumeML +=
    commandedRateMLHr *
    (elapsedMs / 3600000.0);

  if (deliveredVolumeML >= vtbiTargetML) {

    Serial.print("VTBI COMPLETE: delivered ");
    Serial.print(deliveredVolumeML, 2);
    Serial.print(" / ");
    Serial.print(vtbiTargetML, 2);
    Serial.println(
      " mL - stopping pump."
    );

    stepper.setSpeed(0);
    disableDriver();

    pumpShouldRun = false;
    commandedRateMLHr = 0.0;
    vtbiActive = false;

    lcd.clear();

    lcdLine(0, "VTBI COMPLETE");

    lcdLineString(
      1,
      String(deliveredVolumeML, 1) +
      " mL delivered"
    );

    lcdLine(3, "0 = change mode");

    beep(3, 100);
  }
}


// ============================================================
// CALIBRATION
// ============================================================

void startCalibration(long steps) {

  if (steps <= 0) {

    Serial.println(
      "CAL REJECTED: steps must be a positive integer, e.g. CAL:3200"
    );

    return;
  }

  if (emergencyStopLatched) {

    Serial.println(
      "CAL REJECTED: E-STOP active."
    );

    return;
  }

  if (pumpShouldRun) {

    Serial.println(
      "CAL REJECTED: pump is already running - stop it first (# or STOP)."
    );

    return;
  }

  if (endLimitActive()) {

    Serial.println(
      "CAL REJECTED: END limit active - cannot move forward."
    );

    return;
  }

  calibrationActive = true;
  calibrationStepsTarget = steps;
  calibrationStepsDone = 0;

  pumpShouldRun = true;
  manualModeActive = true;
  commandedRateMLHr = CALIBRATION_SPEED_ML_HR;

  enableDriver();

  stepper.setSpeed(
    rateToStepsPerSecond(
      CALIBRATION_SPEED_ML_HR
    )
  );

  Serial.print("CAL: running ");
  Serial.print(steps);
  Serial.println(
    " steps at calibration speed. Do not touch the keypad/PC until it reports done."
  );

  Serial.print(
    "CAL: at the CURRENT STEPS_PER_ML="
  );

  Serial.print(STEPS_PER_ML, 2);

  Serial.print(
    ", expected travel = "
  );

  Serial.print(
    steps / STEPS_PER_ML,
    4
  );

  Serial.println(
    " mL - measure the REAL result once it stops and compare."
  );

  lcd.clear();

  lcdLine(0, "CALIBRATION JOG");

  lcdLineString(
    1,
    String(steps) + " steps"
  );

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

  Serial.println(
    "CAL: now physically measure the actual plunger travel (mm) or expelled volume (mL)."
  );

  Serial.print(
    "CAL: expected at current settings = "
  );

  Serial.print(
    calibrationStepsTarget / STEPS_PER_ML,
    4
  );

  Serial.println(" mL.");

  Serial.println(
    "CAL: if the real number differs, real_STEPS_PER_ML = steps / real_mL - update"
  );

  Serial.println(
    "     MICROSTEPS_PER_TURN (or STEPS_PER_ML directly) in the header to match."
  );

  lcd.clear();

  lcdLine(0, "CAL DONE");

  lcdLineString(
    1,
    String(calibrationStepsTarget) +
    " steps sent"
  );

  lcdLine(2, "Measure now,");
  lcdLine(3, "check Serial");

  beep(2, 100);
}


// ============================================================
// STATUS
// ============================================================

void printStatusLine() {

  Serial.print("STATUS,mode=");

  Serial.print(
    operatingMode == MODE_MANUAL
      ? "MANUAL"
      : operatingMode == MODE_AUTONOMOUS
        ? "AUTONOMOUS"
        : "UNSELECTED"
  );

  Serial.print(",rate=");
  Serial.print(commandedRateMLHr, 2);

  Serial.print(",running=");
  Serial.print(pumpShouldRun ? 1 : 0);

  Serial.print(",estop=");
  Serial.print(emergencyStopLatched ? 1 : 0);

  Serial.print(",home=");
  Serial.print(homeLimitActive() ? 1 : 0);

  Serial.print(",end=");
  Serial.print(endLimitActive() ? 1 : 0);

  Serial.print(",battery_v=");
  Serial.print(smoothedBatteryVoltage, 2);

  Serial.print(",battery_pct=");
  Serial.print(
    batteryPercentFromVoltage(
      smoothedBatteryVoltage
    ),
    0
  );

  Serial.print(",vtbi_active=");
  Serial.print(vtbiActive ? 1 : 0);

  Serial.print(",vtbi_target_ml=");
  Serial.print(vtbiTargetML, 2);

  Serial.print(",vtbi_delivered_ml=");
  Serial.print(deliveredVolumeML, 2);

  Serial.print(",cal_active=");
  Serial.print(calibrationActive ? 1 : 0);

  Serial.print(",cal_steps_done=");
  Serial.print(calibrationStepsDone);

  Serial.print(",cal_steps_target=");
  Serial.print(calibrationStepsTarget);

  Serial.print(",position=");
  Serial.println(stepper.currentPosition());
}


// ============================================================
// SERIAL LINE PROTOCOL
// ============================================================

void handleSerialLine(String line) {

  line.trim();

  if (line.length() == 0) {
    return;
  }

  if (line.startsWith("SET_RATE:")) {

    if (operatingMode == MODE_MANUAL) {

      Serial.println(
        "REJECTED: system is in MANUAL mode - PC control is disabled."
      );

      Serial.println(
        "Press 0 on the keypad (from idle) to switch to AUTONOMOUS mode."
      );

      return;
    }

    if (entryMode != ENTRY_NONE) {
      cancelDoseEntry("PC sent SET_RATE");
    }

    manualModeActive = false;
    vtbiActive = false;
    calibrationActive = false;

    float requested =
      line.substring(9).toFloat();

    lastRateCommandTime = millis();

    applyRate(requested);

    Serial.print("OK,rate=");
    Serial.println(
      commandedRateMLHr,
      2
    );

    lcd.clear();

    lcdLine(0, "PUMP RUNNING");

    lcdLineString(
      1,
      "Rate: " +
      String(commandedRateMLHr, 2) +
      " mL/hr"
    );

    lcdLine(
      2,
      driverEnabled
        ? "Driver: ENABLED"
        : "Driver: DISABLED"
    );

    lcdLine(
      3,
      emergencyStopLatched
        ? "E-STOP ACTIVE"
        : "OK"
    );

    return;
  }

  if (line == "STOP") {

    stepper.setSpeed(0);
    disableDriver();

    pumpShouldRun = false;
    commandedRateMLHr = 0.0;
    vtbiActive = false;
    calibrationActive = false;

    Serial.println("OK,stopped");

    lcdLine(3, "STOPPED (cmd)");

    return;
  }

  if (line == "GET_STATUS") {

    printStatusLine();

    return;
  }

  if (line.startsWith("CAL:")) {

    long steps =
      line.substring(4).toInt();

    startCalibration(steps);

    return;
  }

  if (line.length() == 1) {

    handleKeypadOrManualCommand(
      line.charAt(0)
    );

    return;
  }

  Serial.print("Unrecognized command: ");
  Serial.println(line);
}


// ============================================================
// MODE SELECTION
// ============================================================

void showModeSelectScreen() {

  lcd.clear();

  lcdLine(0, "SELECT MODE:");
  lcdLine(1, "A=Manual(keypad)");
  lcdLine(2, "B=Autonomous(PC)");
  lcdLine(3, "");

  Serial.println(
    "SELECT MODE: send 'A'/'MODE:MANUAL' or 'B'/'MODE:AUTO',"
  );

  Serial.println(
    "or just send any SET_RATE/GET_STATUS to auto-select Autonomous."
  );
}

void selectMode(OperatingMode mode) {

  operatingMode = mode;

  Serial.println(
    mode == MODE_MANUAL
      ? "MODE: MANUAL selected."
      : "MODE: AUTONOMOUS selected. Awaiting SET_RATE from PC."
  );

  beep(1, 80);

  showIdleScreen();
}

void handleModeSelection() {

  char key = getNewKeyPress();

  if (key == 'A' || key == 'a') {

    selectMode(MODE_MANUAL);

    return;
  }

  if (key == 'B' || key == 'b') {

    selectMode(MODE_AUTONOMOUS);

    return;
  }

  while (Serial.available() > 0) {

    char c = Serial.read();

    if (c == '\n') {

      String line =
        serialLineBuffer;

      line.trim();

      serialLineBuffer = "";

      if (line == "MODE:MANUAL") {

        selectMode(MODE_MANUAL);
      }
      else if (
        line == "MODE:AUTO" ||
        line.startsWith("SET_RATE:") ||
        line == "GET_STATUS"
      ) {

        selectMode(MODE_AUTONOMOUS);
      }
    }
    else if (c != '\r') {

      serialLineBuffer += c;
    }
  }
}


// ============================================================
// IDLE SCREEN
// ============================================================

void showIdleScreen() {

  lcd.clear();

  if (emergencyStopLatched) {

    lcdLine(0, "E-STOP ACTIVE");
    lcdLine(3, "Press R to reset");

    return;
  }

  lcdLine(
    0,
    operatingMode == MODE_MANUAL
      ? "MODE: MANUAL"
      : "MODE: AUTONOMOUS"
  );

  if (vtbiActive) {

    lcdLineString(
      1,
      "Rate: " +
      String(commandedRateMLHr, 2) +
      " mL/hr"
    );

    lcdLineString(
      2,
      "VTBI:" +
      String(
        vtbiTargetML - deliveredVolumeML,
        1
      ) +
      "mL left"
    );
  }
  else {

    lcdLineString(
      1,
      "Rate: " +
      String(commandedRateMLHr, 2) +
      " mL/hr"
    );

    lcdLine(
      2,
      driverEnabled
        ? "Driver: ENABLED"
        : "Driver: DISABLED"
    );
  }

  lcdLine(
    3,
    operatingMode == MODE_MANUAL
      ? "A=dose 0=mode"
      : "0 = change mode"
  );
}


// ============================================================
// MANUAL DOSE ENTRY
// ============================================================

void startDoseEntry() {

  if (emergencyStopLatched) {

    Serial.println(
      "Cannot start dose entry - E-STOP active."
    );

    beep(2, 70);

    return;
  }

  if (calibrationActive) {

    Serial.println(
      "Cannot start dose entry - a calibration jog is running. Wait for it to finish."
    );

    beep(2, 70);

    return;
  }

  if (operatingMode != MODE_MANUAL) {

    Serial.println(
      "Cannot start dose entry - system is in AUTONOMOUS mode. Press 0 to change mode."
    );

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

void backspaceEntryBuffer(byte displayRow) {

  if (entryBuffer.length() > 0) {
    entryBuffer.remove(
      entryBuffer.length() - 1
    );
  }

  lcdLineString(
    displayRow,
    entryBuffer
  );
}


// ============================================================
// HANDLE ENTRY KEY
// ============================================================

void handleEntryKey(char key) {

  if (key == 'D' || key == 'd') {

    cancelDoseEntry(
      "user pressed D"
    );

    return;
  }

  // -------------------- PROGRAM SELECT --------------------

  if (entryMode == ENTRY_PROGRAM_SELECT) {

    if (key == '1') {

      selectedProgram =
        PROGRAM_RATE_VTBI;

      entryMode = ENTRY_RATE;
      entryBuffer = "";

      lcd.clear();

      lcdLine(0, "RATE + VTBI");
      lcdLine(1, "Rate (mL/hr):");
      lcdLine(3, "#nxt D=can C=del");

      return;
    }

    if (key == '2') {

      selectedProgram =
        PROGRAM_WEIGHT_DOSE;

      entryMode = ENTRY_WEIGHT;
      entryBuffer = "";

      lcd.clear();

      lcdLine(0, "WEIGHT DOSE");
      lcdLine(1, "Weight (kg):");
      lcdLine(3, "#nxt D=can C=del");

      return;
    }

    if (key == '3') {

      selectedProgram =
        PROGRAM_VOLUME_TIME;

      entryMode = ENTRY_VT_VOLUME;
      entryBuffer = "";

      lcd.clear();

      lcdLine(0, "VOLUME+TIME");
      lcdLine(1, "Volume (mL):");
      lcdLine(3, "D = cancel");

      return;
    }

    return;
  }


  // -------------------- WEIGHT --------------------

  if (entryMode == ENTRY_WEIGHT) {

    if (key == '#') {

      float weight =
        entryBuffer.toFloat();

      if (
        weight <= 0.0 ||
        weight > 300.0
      ) {

        Serial.println(
          "Invalid weight - re-enter."
        );

        lcdLine(
          2,
          "Invalid! Retry:"
        );

        entryBuffer = "";

        beep(2, 70);

        return;
      }

      enteredWeightKg = weight;
      entryBuffer = "";
      entryMode = ENTRY_UNIT_SELECT;

      lcd.clear();

      lcdLineString(
        0,
        "Weight: " +
        String(
          enteredWeightKg,
          1
        ) +
        " kg"
      );

      lcdLine(1, "B=mL/hr");
      lcdLine(2, "C=ug/kg/min");
      lcdLine(3, "D=cancel");

      return;
    }

    if (key == 'C' || key == 'c') {

      backspaceEntryBuffer(2);

      return;
    }

    if (
      (key >= '0' && key <= '9') ||
      key == '*'
    ) {

      entryBuffer +=
        (key == '*') ? '.' : key;

      lcdLineString(
        2,
        entryBuffer
      );
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
      lcdLine(3, "#OK D=canc C=del");

      return;
    }

    if (key == 'C' || key == 'c') {

      enteredUnitIsMcgKgMin = true;
      entryMode = ENTRY_RATE;
      entryBuffer = "";

      lcd.clear();

      lcdLine(
        0,
        "Rate(ug/kg/min):"
      );

      lcdLine(
        3,
        "#OK D=canc C=del"
      );

      return;
    }

    return;
  }


  // -------------------- RATE --------------------

  if (entryMode == ENTRY_RATE) {

    if (key == '#') {

      float enteredValue =
        entryBuffer.toFloat();

      if (enteredValue < 0.0) {

        Serial.println(
          "Invalid rate - re-enter."
        );

        lcdLine(
          1,
          "Invalid! Retry:"
        );

        entryBuffer = "";

        beep(2, 70);

        return;
      }

      if (
        selectedProgram ==
        PROGRAM_RATE_VTBI
      ) {

        pendingRateMLHr =
          enteredValue;

        entryBuffer = "";
        entryMode = ENTRY_VTBI;

        lcd.clear();

        lcdLineString(
          0,
          "Rate: " +
          String(
            pendingRateMLHr,
            2
          ) +
          " mL/hr"
        );

        lcdLine(1, "VTBI (mL):");
        lcdLine(3, "#OK D=canc C=del");

        return;
      }

      float rateMLHr =
        enteredUnitIsMcgKgMin
          ? (
              enteredValue *
              enteredWeightKg *
              60.0
            ) /
            DRUG_CONCENTRATION_MCG_ML
          : enteredValue;

      entryMode = ENTRY_NONE;
      entryBuffer = "";
      manualModeActive = true;

      applyRate(rateMLHr);

      Serial.print(
        "MANUAL dose set: weight="
      );

      Serial.print(
        enteredWeightKg,
        1
      );

      Serial.print(
        "kg, entered="
      );

      Serial.print(
        enteredValue,
        3
      );

      Serial.println(
        enteredUnitIsMcgKgMin
          ? " ug/kg/min"
          : " mL/hr"
      );

      Serial.print(
        "-> commanded rate = "
      );

      Serial.print(
        commandedRateMLHr,
        2
      );

      Serial.println(
        " mL/hr"
      );

      lcd.clear();

      lcdLine(
        0,
        "MANUAL MODE"
      );

      lcdLineString(
        1,
        "Rate: " +
        String(
          commandedRateMLHr,
          2
        ) +
        " mL/hr"
      );

      lcdLineString(
        2,
        "Weight: " +
        String(
          enteredWeightKg,
          1
        ) +
        " kg"
      );

      lcdLine(
        3,
        driverEnabled
          ? "Driver: ENABLED"
          : "Driver: DISABLED"
      );

      beep(2, 80);

      return;
    }

    if (key == 'C' || key == 'c') {

      backspaceEntryBuffer(1);

      return;
    }

    if (
      (key >= '0' && key <= '9') ||
      key == '*'
    ) {

      entryBuffer +=
        (key == '*') ? '.' : key;

      lcdLineString(
        1,
        entryBuffer
      );
    }

    return;
  }


  // -------------------- VTBI --------------------

  if (entryMode == ENTRY_VTBI) {

    if (key == '#') {

      float enteredVtbi =
        entryBuffer.toFloat();

      if (enteredVtbi <= 0.0) {

        Serial.println(
          "Invalid VTBI - re-enter."
        );

        lcdLine(
          2,
          "Invalid! Retry:"
        );

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

      applyRate(
        pendingRateMLHr
      );

      Serial.print(
        "RATE+VTBI set: rate="
      );

      Serial.print(
        commandedRateMLHr,
        2
      );

      Serial.print(
        " mL/hr, VTBI="
      );

      Serial.print(
        vtbiTargetML,
        2
      );

      Serial.println(" mL");

      lcd.clear();

      lcdLine(
        0,
        "RATE+VTBI MODE"
      );

      lcdLineString(
        1,
        "Rate: " +
        String(
          commandedRateMLHr,
          2
        ) +
        " mL/hr"
      );

      lcdLineString(
        2,
        "VTBI:" +
        String(
          vtbiTargetML,
          1
        ) +
        "mL left"
      );

      lcdLine(
        3,
        driverEnabled
          ? "Driver: ENABLED"
          : "Driver: DISABLED"
      );

      beep(2, 80);

      return;
    }

    if (key == 'C' || key == 'c') {

      backspaceEntryBuffer(2);

      return;
    }

    if (
      (key >= '0' && key <= '9') ||
      key == '*'
    ) {

      entryBuffer +=
        (key == '*') ? '.' : key;

      lcdLineString(
        2,
        entryBuffer
      );
    }

    return;
  }


  // -------------------- VOLUME+TIME: VOLUME --------------------

  if (entryMode == ENTRY_VT_VOLUME) {

    if (key == '#') {

      float enteredVolume =
        entryBuffer.toFloat();

      if (enteredVolume <= 0.0) {

        Serial.println(
          "Invalid volume - re-enter."
        );

        lcdLine(
          2,
          "Invalid! Retry:"
        );

        entryBuffer = "";

        beep(2, 70);

        return;
      }

      pendingVolumeML =
        enteredVolume;

      entryBuffer = "";
      entryMode = ENTRY_VT_TIME;

      lcd.clear();

      lcdLineString(
        0,
        "Volume: " +
        String(
          pendingVolumeML,
          2
        ) +
        " mL"
      );

      lcdLine(
        1,
        "Time (min):"
      );

      lcdLine(
        3,
        "#OK D=canc C=del"
      );

      return;
    }

    if (key == 'C' || key == 'c') {

      backspaceEntryBuffer(2);

      return;
    }

    if (
      (key >= '0' && key <= '9') ||
      key == '*'
    ) {

      entryBuffer +=
        (key == '*') ? '.' : key;

      lcdLineString(
        2,
        entryBuffer
      );
    }

    return;
  }


  // -------------------- VOLUME+TIME: TIME --------------------

  if (entryMode == ENTRY_VT_TIME) {

    if (key == '#') {

      float enteredMinutes =
        entryBuffer.toFloat();

      if (enteredMinutes <= 0.0) {

        Serial.println(
          "Invalid time - re-enter."
        );

        lcdLine(
          2,
          "Invalid! Retry:"
        );

        entryBuffer = "";

        beep(2, 70);

        return;
      }

      float computedRateMLHr =
        pendingVolumeML /
        (enteredMinutes / 60.0);

      entryMode = ENTRY_NONE;
      entryBuffer = "";
      manualModeActive = true;

      vtbiActive = true;
      vtbiTargetML = pendingVolumeML;
      deliveredVolumeML = 0.0;
      lastVtbiUpdateTime = millis();

      applyRate(
        computedRateMLHr
      );

      Serial.print(
        "VOLUME+TIME set: volume="
      );

      Serial.print(
        pendingVolumeML,
        2
      );

      Serial.print(
        " mL over "
      );

      Serial.print(
        enteredMinutes,
        1
      );

      Serial.print(
        " min -> rate="
      );

      Serial.print(
        commandedRateMLHr,
        2
      );

      Serial.println(
        " mL/hr"
      );

      lcd.clear();

      lcdLine(
        0,
        "VOL+TIME MODE"
      );

      lcdLineString(
        1,
        "Rate: " +
        String(
          commandedRateMLHr,
          2
        ) +
        " mL/hr"
      );

      lcdLineString(
        2,
        String(
          pendingVolumeML,
          1
        ) +
        "mL/" +
        String(
          enteredMinutes,
          0
        ) +
        "min"
      );

      lcdLine(
        3,
        driverEnabled
          ? "Driver: ENABLED"
          : "Driver: DISABLED"
      );

      beep(2, 80);

      return;
    }

    if (key == 'C' || key == 'c') {

      backspaceEntryBuffer(2);

      return;
    }

    if (
      (key >= '0' && key <= '9') ||
      key == '*'
    ) {

      entryBuffer +=
        (key == '*') ? '.' : key;

      lcdLineString(
        2,
        entryBuffer
      );
    }

    return;
  }
}


// ============================================================
// KEYPAD / MANUAL COMMANDS
// ============================================================

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

      showBatteryOnLCD();

      break;


    case '#':

      stepper.setSpeed(0);
      disableDriver();

      pumpShouldRun = false;
      commandedRateMLHr = 0.0;
      manualModeActive = false;
      vtbiActive = false;
      calibrationActive = false;

      Serial.println(
        "Pump stopped (keypad #)."
      );

      showIdleScreen();

      beep(1, 80);

      break;


    case '0':

      if (pumpShouldRun) {

        Serial.println(
          "Cannot change mode while the pump is running - stop it first (# or STOP)."
        );

        beep(2, 70);

        break;
      }

      operatingMode =
        MODE_UNSELECTED;

      Serial.println(
        "Returning to mode selection."
      );

      showModeSelectScreen();

      break;


    default:

      Serial.print(
        "Key/char received (no automated action bound): "
      );

      Serial.println(cmd);

      break;
  }
}


// ============================================================
// I2C SCANNER
// ============================================================

void scanI2C() {

  Serial.println(
    "Scanning I2C bus..."
  );

  byte count = 0;

  for (
    byte address = 1;
    address < 127;
    address++
  ) {

    Wire.beginTransmission(address);

    if (
      Wire.endTransmission() == 0
    ) {

      Serial.print(
        "  Found I2C device at 0x"
      );

      Serial.println(
        address,
        HEX
      );

      count++;
    }
  }

  if (count == 0) {

    Serial.println(
      "  No I2C devices found - check LCD wiring/power."
    );
  }
}


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  //Serial.setTxTimeoutMs(0);

  delay(500);

  Serial.println();
  Serial.println(
    "=========================================="
  );

  Serial.println(
    " SMART SYRINGE INFUSION PUMP - REAL FIRMWARE V1"
  );

  Serial.println(
    " ESP32-S3"
  );

  Serial.println(
    "=========================================="
  );

  Serial.println();

  Serial.print(
    "Derived STEPS_PER_ML = "
  );

  Serial.println(
    STEPS_PER_ML,
    2
  );

  Serial.println(
    "(verify against your board's actual microstepping setting - see header comment)"
  );

  Serial.println();


  // -------------------- SAFETY INPUTS --------------------

  pinMode(
    E_STOP_PIN,
    INPUT_PULLUP
  );

  pinMode(
    HOME_LIMIT_PIN,
    INPUT_PULLUP
  );

  pinMode(
    END_LIMIT_PIN,
    INPUT_PULLUP
  );


  // -------------------- BUZZER --------------------

  pinMode(
    BUZZER_PIN,
    OUTPUT
  );

  digitalWrite(
    BUZZER_PIN,
    LOW
  );


  // -------------------- STATUS LEDS --------------------

  pinMode(
    GREEN_LED_PIN,
    OUTPUT
  );

  pinMode(
    RED_LED_PIN,
    OUTPUT
  );

  digitalWrite(
    GREEN_LED_PIN,
    LOW
  );

  digitalWrite(
    RED_LED_PIN,
    LOW
  );


  // -------------------- BATTERY ADC --------------------

  analogReadResolution(12);

  analogSetPinAttenuation(
    BATTERY_ADC_PIN,
    ADC_11db
  );

  updateBatteryReading();


  // -------------------- A4988 --------------------

  pinMode(
    EN_PIN,
    OUTPUT
  );

  digitalWrite(
    EN_PIN,
    DRIVER_DISABLED
  );

  driverEnabled = false;


  // -------------------- I2C + LCD --------------------

  Wire.begin(
    SDA_PIN,
    SCL_PIN
  );

  scanI2C();

  lcd.begin(16, 4);

  lcd.setBacklight(255);

  lcd.clear();

  lcdLine(
    0,
    "SMART PUMP"
  );

  lcdLine(
    1,
    "Real Firmware V1"
  );

  lcdLine(
    2,
    "Initializing..."
  );


  // -------------------- STEPPER --------------------

  stepper.setPinsInverted(
    false,
    false,
    false
  );

  stepper.setMaxSpeed(
    rateToStepsPerSecond(
      MAX_FLOW_ML_HR
    ) * 1.1
  );

  stepper.setAcceleration(
    MOTOR_ACCELERATION_STEPS_S2
  );

  stepper.setSpeed(0);


  // -------------------- INITIAL E-STOP CHECK --------------------

  if (
    digitalRead(E_STOP_PIN) == LOW
  ) {

    emergencyStopLatched = true;

    lcd.clear();

    lcdLine(
      0,
      "E-STOP PRESSED"
    );

    lcdLine(
      1,
      "Pump disabled"
    );

    lcdLine(
      2,
      "Release E-STOP"
    );

    lcdLine(
      3,
      "Press R to reset"
    );
  }
  else {

    beep(2, 100);

    showModeSelectScreen();
  }

  lastRateCommandTime = millis();

  Serial.println(
    "Setup complete."
  );

  Serial.println();
}


// ============================================================
// MAIN LOOP
// ============================================================

void loop() {

  // 1. E-STOP
  checkEmergencyStop();

  // Buzzer
  updateBuzzer();

  // 1b. Mode selection
  if (
    operatingMode ==
    MODE_UNSELECTED
  ) {

    handleModeSelection();

    return;
  }

  // 2. Limit enforcement
  enforceLimits();

  // 3. Watchdog
  checkWatchdog();

  // 3b. VTBI
  updateVtbiTracking();

  // 4. Status LEDs
  updateLeds();

  // 5. Battery
  if (
    millis() - lastBatteryReport >=
    BATTERY_REPORT_INTERVAL_MS
  ) {

    updateBatteryReading();

    lastBatteryReport = millis();
  }

  // 6. Keypad
  char key =
    getNewKeyPress();

  if (key) {

    if (
      entryMode != ENTRY_NONE
    ) {

      if (emergencyStopLatched) {

        cancelDoseEntry(
          "E-STOP activated"
        );
      }
      else {

        handleEntryKey(key);
      }
    }
    else if (
      !emergencyStopLatched ||
      key == 'R' ||
      key == 'r'
    ) {

      handleKeypadOrManualCommand(
        key
      );
    }
    else {

      Serial.println(
        "Key ignored - E-STOP active. Press R."
      );
    }
  }


  // 7. Serial line protocol
  while (
    Serial.available() > 0
  ) {

    char c =
      Serial.read();

    if (c == '\n') {

      if (
        !emergencyStopLatched ||
        serialLineBuffer == "R" ||
        serialLineBuffer == "r"
      ) {

        handleSerialLine(
          serialLineBuffer
        );
      }
      else {

        Serial.println(
          "Command ignored - E-STOP active. Send R."
        );
      }

      serialLineBuffer = "";
    }
    else if (c != '\r') {

      serialLineBuffer += c;
    }
  }


  // 8. Run stepper
  if (calibrationActive) {

    if (
      stepper.runSpeed()
    ) {

      calibrationStepsDone++;

      if (
        calibrationStepsDone >=
        calibrationStepsTarget
      ) {

        finishCalibration();
      }
    }
  }
  else {

    stepper.runSpeed();
  }
}