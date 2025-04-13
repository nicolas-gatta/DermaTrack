import serial
import json

ser = serial.Serial(port = 'COM5', baudrate=115200)
 
while True:
    line = ser.readline().decode('utf-8').strip()
    if line:
        try:
            data = json.loads(line)
            print("Received:", data)
            
            if(data["take_picture"]):
                payload = {"picture_taken": True}
                json_str = json.dumps(payload)
                ser.write(json_str.encode('utf-8'))
                print("Data Sended !")
                break

        except json.JSONDecodeError:
            print("Received:", line)
