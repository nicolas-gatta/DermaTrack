#define ARDUINOJSON_SLOT_ID_SIZE 1
#define ARDUINOJSON_STRING_LENGTH_SIZE 1
#define ARDUINOJSON_USE_DOUBLE 0
#define ARDUINOJSON_USE_LONG_LONG 0

#include <Wire.h>
#include <ArduinoJson.h>
#include <SparkFun_VL53L5CX_Library.h>

SparkFun_VL53L5CX myImager;
VL53L5CX_ResultsData measurementData;

JsonDocument data;
int sensorResolution = 0;
int numberDataPerRow = 0;

void setup()
{
  Serial.begin(115200);
  delay(1000);

  Wire.begin(); //This resets to 100kHz I2C
  Wire.setClock(400000); //Sensor has max I2C freq of 400kHz 
  
  Serial.println("Initializing sensor board. This can take up to 10s. Please wait.");
  if (myImager.begin() == false){
    Serial.println(F("Sensor not found - check your wiring. Freezing"));
    while (1) ;
  }
  
  myImager.setResolution(8*8); //Enable all 64 pads

  sensorResolution = myImager.getResolution();
  numberDataPerRow = sqrt(sensorResolution);

  myImager.startRanging();

  
}

void loop()
{
  //Poll sensor for new data
  if (myImager.isDataReady() == true){
    if (myImager.getRangingData(&measurementData)){
      JsonArray distances = data["distances"].to<JsonArray>();
      for (int x = 0 ; x < sensorResolution; x += numberDataPerRow){
        JsonArray data_0 = distances.add<JsonArray>();
        for (int y = 0; y < numberDataPerRow; y++){
          data_0.add(measurementData.distance_mm[x + y]);
        }
      }
      serializeJson(data, Serial);
      Serial.println();
      Serial.println();
    }
  }

  delay(5);
}
