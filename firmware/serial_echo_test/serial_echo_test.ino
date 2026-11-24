// Minimal Serial RX/TX test - completely independent of pump_firmware.ino.
// Purpose: find out whether this board's Serial RX works AT ALL, with
// zero other code (no keypad, no stepper, no LCD) that could possibly
// interfere.
//
// HOW TO USE:
//   1. Upload this sketch (same board/port settings you've been using:
//      USB CDC On Boot = Enabled, same USB Mode setting as before).
//   2. Open Serial Monitor, baud 115200, line ending = Newline.
//   3. Type "hello" and press Enter/Send.
//
// EXPECTED IF WORKING:
//   You should see this printed roughly once a second:
//     "alive, waiting for input..."
//   And the moment you send "hello", you should immediately see:
//     "GOT IT: hello"
//
// IF "alive, waiting for input..." NEVER APPEARS AT ALL:
//   TX itself is broken too - different problem than what we've been
//   chasing (go back to checking the board/cable/port selection).
//
// IF "alive, waiting..." appears fine but "GOT IT:" NEVER appears no
// matter what you send:
//   This conclusively proves RX is broken at the board/cable/driver
//   level, completely unrelated to pump_firmware.ino's own code - the
//   fix at that point is hardware-side (try a different USB cable
//   first - this is the single most common real-world cause of
//   exactly this TX-works/RX-doesn't pattern - then a different USB
//   port, then consider the board itself may have a damaged RX line).

unsigned long lastPing = 0;

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("=== Minimal Serial echo test ===");
  Serial.println("Type anything and press Enter - watch for 'GOT IT:' below.");
}

void loop() {
  if (millis() - lastPing >= 1000) {
    lastPing = millis();
    Serial.println("alive, waiting for input...");
  }

  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      Serial.print("GOT IT: ");
      Serial.println(line);
    }
  }
}
