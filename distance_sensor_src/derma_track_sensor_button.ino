#define ARDUINOJSON_SLOT_ID_SIZE 1
#define ARDUINOJSON_STRING_LENGTH_SIZE 1
#define ARDUINOJSON_USE_DOUBLE 0
#define ARDUINOJSON_USE_LONG_LONG 0

#include <ArduinoJson.h>
#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

//-----------------Sensor Variables
SparkFun_VL53L5CX myImager;
VL53L5CX_ResultsData measurementData;
int sensorResolution = 0;
int numberDataPerRow = 0;

//-----------------Button Variables
int BUTTON_PIN = 12;
int state = LOW;
int reading;
int previous = HIGH;

//-----------------Json Variable
JsonDocument dataSend;
JsonDocument dataReceive;


void check_button_connection(){
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    if(digitalRead(BUTTON_PIN) == LOW){
        Serial.println(F("Button not found - check wiring. Freezing...."));
        while(1);
    }
}

void check_sensor_connection(){
    if (myImager.begin() == false){
        Serial.println(F("Sensor not found - check your wiring. Freezing"));
        while (1) ;
    }
}

void setup() {
    
    Serial.begin(115200);
    Serial.println("Initializing sensor board. This can take up to 10s. Please wait.");

    check_button_connection();
    
    dataSend["take_picture"] = false;
    dataSend["distances"] = null;
    dataReceive["picture_taken"] = false;

    Wire.begin(); //This resets to 100kHz I2C
    Wire.setClock(400000); //Sensor has max I2C freq of 400kHz 
    
    check_sensor_connection();
    
    myImager.setResolution(8*8);

    sensorResolution = myImager.getResolution();
    numberDataPerRow = sqrt(sensorResolution);

    myImager.startRanging();
}

void loop(){

  reading = digitalRead(BUTTON_PIN);


  if (!dataSend["take_picture"] && reading == LOW && previous == HIGH){
      dataSend["take_picture"] = true;
  }

  if (dataSend["take_picture"] && myImager.isDataReady()){
    if (myImager.getRangingData(&measurementData)){
      JsonArray distances = dataSend["distances"].to<JsonArray>();
      for (int x = 0 ; x < sensorResolution; x += numberDataPerRow){
        JsonArray data_0 = distances.add<JsonArray>();
        for (int y = 0; y < numberDataPerRow; y++){
          data_0.add(measurementData.distance_mm[x + y]);
        }
      }
    }
  }

  serializeJson(dataSend, Serial);
  Serial.println();
  Serial.println();

  previous = reading;

  if (dataSend["take_picture"]){
    DeserializationError error = deserializeJson(dataReceive, Serial);

    if(!error && dataReceive["picture_taken"]){
        dataSend["take_picture"] = false;
        dataSend["distances"] = null;
        dataReceive["picture_taken"] = false;
    }
  }

}