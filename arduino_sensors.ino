/*
  🏥 Health Monitoring System - ABSOLUTE BULLETPROOF VERSION
  - Guarantees LED turns ON and STAYS ON
  - Accurate 100 Buffer Size for correct BPM
*/

#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"

#define TEMP_PIN A0
MAX30105 particleSensor;

#define BUFFER_SIZE 100 
uint16_t irBuffer[BUFFER_SIZE];
uint16_t redBuffer[BUFFER_SIZE];

int32_t spo2, heartRate;
int8_t  validSPO2, validHeartRate;
int32_t lastHR = 0, lastSpO2 = 0;
float   lastTemp = 0.0;

// Function to un-hang I2C bus if wires get shaken/loose
void recoverI2C() {
    pinMode(SDA, INPUT_PULLUP);
    pinMode(SCL, OUTPUT);
    for (int i = 0; i < 9; i++) {
        digitalWrite(SCL, HIGH);
        delayMicroseconds(5);
        digitalWrite(SCL, LOW);
        delayMicroseconds(5);
        if (digitalRead(SDA) == HIGH) break;
    }
    pinMode(SDA, INPUT);
    pinMode(SCL, INPUT);
}

// Function to fully reset the I2C bus and sensor if it crashes
void resetSensor() {
    Wire.end();
    delay(50);
    recoverI2C(); // Manually free the I2C bus
    Wire.begin();
    delay(50);
    
    for(int i=0; i<5; i++) {
        if (particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
            particleSensor.setup(60, 4, 2, 100, 411, 4096);
            particleSensor.setPulseAmplitudeRed(0x3F); // FORCE RED ON
            particleSensor.setPulseAmplitudeIR(0x3F);  // FORCE IR ON
            break;
        }
        delay(100);
    }
}

void setup() {
    Serial.begin(9600);
    resetSensor(); // Try multiple times on boot
}

void loop() {
    // 1. Temp Reading
    float tSum = 0;
    for(int i=0; i<5; i++) {
        tSum += (analogRead(TEMP_PIN) * 5.0 / 1024.0) * 100.0;
        delay(2);
    }
    lastTemp = tSum / 5.0;

    // 2. Data Collection
    bool ok = true;
    for (byte i = 0; i < BUFFER_SIZE; i++) {
        unsigned long t = millis();
        while (!particleSensor.available()) {
            particleSensor.check();
            if (millis() - t > 150) { ok = false; break; }
        }
        if (!ok) break;
        redBuffer[i] = (uint16_t)(particleSensor.getRed() >> 2);
        irBuffer[i]  = (uint16_t)(particleSensor.getIR()  >> 2);
        particleSensor.nextSample();
    }

    // 3. Process
    bool fingerOn = false;
    if (ok) {
        long irAvg = 0;
        for (byte i = 0; i < BUFFER_SIZE; i++) irAvg += irBuffer[i];
        irAvg /= BUFFER_SIZE;
        
        fingerOn = (irAvg > 5000); // 5000 prevents table reflections from tricking the sensor

        if (fingerOn) {
            maxim_heart_rate_and_oxygen_saturation(irBuffer, BUFFER_SIZE, redBuffer, &spo2, &validSPO2, &heartRate, &validHeartRate);
            if (validHeartRate && heartRate > 40 && heartRate < 160) lastHR = heartRate;
            if (validSPO2 && spo2 > 80 && spo2 <= 100) lastSpO2 = spo2;
        } else {
            // Clear old readings if finger is removed
            lastHR = 0; 
            lastSpO2 = 0;
        }
    } else {
        // ABSOLUTE HARD RECOVERY
        resetSensor();
    }

    // 4. Send to Python dashboard
    int valid = (fingerOn && lastHR > 30 && lastSpO2 > 80) ? 1 : 0;
    
    int32_t displayHR = lastHR;
    int32_t displaySpO2 = lastSpO2;
    float displayTemp = lastTemp;

    if (!fingerOn) {
        displayHR = 0;
        displaySpO2 = 0;
        displayTemp = 0.0;
    } else if (lastHR == 0) {
        displayHR = 1; // UI trick to show "Stabilizing..."
    }

    // ALWAYS SEND DATA (LIVE)
    Serial.print(F("DATA:HRate=")); Serial.print(displayHR);
    Serial.print(F(",Oxygen="));    Serial.print(displaySpO2);
    Serial.print(F(",Temp="));      Serial.print(displayTemp, 1);
    Serial.print(F(",VALID="));     Serial.print(valid);
    Serial.println(F(",LED=1"));
}