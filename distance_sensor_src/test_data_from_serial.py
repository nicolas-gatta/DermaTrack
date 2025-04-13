import serial
import json

ser = serial.Serial(port = 'COM5', baudrate=115200)
 
while True:
    line = ser.readline().decode('utf-8').strip()
    if line:
        try:
            data = json.loads(line)
            print("Received:", data)
        except json.JSONDecodeError:
            print("Received:", line)
