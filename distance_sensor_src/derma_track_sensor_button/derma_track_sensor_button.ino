#define ARDUINOJSON_SLOT_ID_SIZE 1
#define ARDUINOJSON_STRING_LENGTH_SIZE 1
#define ARDUINOJSON_USE_DOUBLE 0
#define ARDUINOJSON_USE_LONG_LONG 0

#include <ArduinoJson.h>
#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

//-----------------Laser Sensor Variables
SparkFun_VL53L5CX myImager;
VL53L5CX_ResultsData measurementData;
int sensorResolution = 0;
int numberDataPerRow = 0;
int mapping_size = 8;
bool is_laser_sensor = true;

//----------------Sonic Sensor Variables
const int TRIGGER_PIN = 27;  
const int ECHO_PIN = 14; 
float duration = 0;

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
    
    is_laser_sensor = false;
    pinMode(TRIGGER_PIN, OUTPUT);  
    pinMode(ECHO_PIN, INPUT_PULLDOWN);
    measure_distance_sonic();
    
    if(dataSend["distances"] == 0){
      Serial.println(F("Sensor not found - check your wiring. Freezing"));
      while(1);
    }
  }else{
    myImager.setResolution(mapping_size*mapping_size);
    sensorResolution = myImager.getResolution();
    numberDataPerRow = sqrt(sensorResolution);
    myImager.startRanging();
  }
  dataSend["distances"] = NULL;
}

void measure_distance_laser(){
  JsonArray distances = dataSend["distances"].to<JsonArray>();
  for (int x = 0 ; x < sensorResolution; x += numberDataPerRow){
    JsonArray data_0 = distances.add<JsonArray>();
    for (int y = 0; y < numberDataPerRow; y++){
      data_0.add(measurementData.distance_mm[x + y]);
    }
  }
}

void measure_distance_sonic(){
  digitalWrite(TRIGGER_PIN, LOW);  
  delayMicroseconds(2);  
  digitalWrite(TRIGGER_PIN, HIGH);  
  delayMicroseconds(10);  
  digitalWrite(TRIGGER_PIN, LOW);  
  duration = pulseIn(ECHO_PIN, HIGH, 40000);
  dataSend["distances"] = (duration * 0.343) / 2;  //2,91545 is the speed of sound in miliemeter per microseconds
}

void setup() {
    
  Serial.begin(115200);
  Serial.println("Initializing sensor board. This can take up to 10s. Please wait.");

  check_button_connection();
  
  dataSend["take_picture"] = false;
  dataSend["distances"] = NULL;
  dataReceive["picture_taken"] = false;

  Wire.begin(); //This resets to 100kHz I2C
  Wire.setClock(400000); //Sensor has max I2C freq of 400kHz 
  
  check_sensor_connection();

  Serial.println(F("Divice Ready to use and sending information at the COM port "));
  Serial.print(Serial);
}

void loop(){

  reading = digitalRead(BUTTON_PIN);


  if (!dataSend["take_picture"] && reading == LOW && previous == HIGH){
    dataSend["take_picture"] = true;
  }

  if (is_laser_sensor && dataSend["take_picture"] && myImager.isDataReady() && myImager.getRangingData(&measurementData)){
    measure_distance_laser();
    
  }else if (!is_laser_sensor && dataSend["take_picture"]) {
    measure_distance_sonic();
  }

  serializeJson(dataSend, Serial);
  Serial.println();
  Serial.println();

  previous = reading;

  if (dataSend["take_picture"]){
    DeserializationError error = deserializeJson(dataReceive, Serial);

    if(!error && dataReceive["picture_taken"]){
        dataSend["take_picture"] = false;
        dataSend["distances"] = NULL;
        dataReceive["picture_taken"] = false;
    }
  }

}