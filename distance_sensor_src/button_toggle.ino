int BUTTON_PIN = 12;
int state = LOW;
int reading;
int previous = HIGH;

void setup() {
   Serial.begin(9600);
   // Define pin #12 as input and activate the internal pull-up resistor
   pinMode(BUTTON_PIN, INPUT_PULLUP);
}

void loop(){
  reading = digitalRead(BUTTON_PIN);

  if (reading == LOW && previous == HIGH){
    if (state == HIGH){
      state = LOW;
    }else{
      state = HIGH;
    }
  }

  Serial.println(state);

  previous = reading;
}