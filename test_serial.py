import serial
import sys
import time

try:
    ser = serial.Serial('COM7', 9600, timeout=2)
    print("Connected to COM7", flush=True)
    for _ in range(5):
        line = ser.readline()
        if line:
            print("READ:", line.decode('utf-8', errors='ignore').strip(), flush=True)
        else:
            print("Timeout reading line", flush=True)
    ser.close()
except Exception as e:
    print("Error:", e, flush=True)
