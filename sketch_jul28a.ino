// Pin Definitions
const int MIC_PIN = 34; // Analog input pin for MAX4466 OUT
const int LED_PIN = 18; // Digital output pin for LED

// Adjust this threshold based on your room's ambient noise level.
// Baseline in silence is ~2048. Audio causes peaks above and dips below.
const int SILENCE_BASELINE = 2048; 
const int THRESHOLD = 300; // Trigger when sound deviates by this much

void setup() {
  Serial.begin(115200);
  
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); // Start with LED off
  
  // Set ADC resolution to 12 bits (0 - 4095 range)
  analogReadResolution(12);
}

void loop() {
  int rawValue = analogRead(MIC_PIN);
  
  // Calculate how far the current reading is from the silent midpoint
  int soundDeviation = abs(rawValue - SILENCE_BASELINE);

  // Print raw value to serial plotter/monitor for easy debugging
  Serial.print("Raw: ");
  Serial.print(rawValue);
  Serial.print(" | Deviation: ");
  Serial.println(soundDeviation);

  // If the sound wave creates a peak/trough wider than our threshold, turn LED ON
  if (soundDeviation > THRESHOLD) {
    digitalWrite(LED_PIN, HIGH);
    delay(50); // Keep LED lit briefly so it's visibly noticeable
  } else {
    digitalWrite(LED_PIN, LOW);
  }

  delay(10); // Short sample delay
}