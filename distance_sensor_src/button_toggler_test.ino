#define ARDUINOJSON_SLOT_ID_SIZE 1
#define ARDUINOJSON_STRING_LENGTH_SIZE 1
#define ARDUINOJSON_USE_DOUBLE 0
#define ARDUINOJSON_USE_LONG_LONG 0

#include <ArduinoJson.h>

int BUTTON_PIN = 12;
int state = LOW;
int reading;
int previous = HIGH;
JsonDocument data;

void setup() {
   
   Serial.begin(115200);

   pinMode(BUTTON_PIN, INPUT_PULLUP);

   data["take_picture"] = false;

}

void loop(){
  
  reading = digitalRead(BUTTON_PIN);

  if (reading == LOW && previous == HIGH){
    if (state == HIGH){
      data["take_picture"] = false;
      state = LOW;
    }else{
      data["take_picture"] = true;
      state = HIGH;
    }
  }

  serializeJson(data, Serial);
  Serial.println();
  Serial.println();

  previous = reading;
}