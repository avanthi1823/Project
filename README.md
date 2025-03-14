from flask import Flask, Response, render_template
import cv2
import numpy as np
app = Flask(_name_)
# Load MobileNet SSD model
prototxt_path = "models/MobileNetSSD_deploy.prototxt.txt"
model_path = "models/MobileNetSSD_deploy.caffemodel"
  net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# Define class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Initialize Video Capture
cap = cv2.VideoCapture(0)

def generate_frames(detect=False):
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            if detect:
                # Prepare frame for detection
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:  # Detection threshold
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        label = f"{CLASSES[idx]}: {confidence*100:.2f}%"
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(frame, label, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """ Main page with links to video streams """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """ Normal video streaming route """
    return Response(generate_frames(detect=False), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_objects')
def detect_objects():
    """ Object detection video streaming route """
    return Response(generate_frames(detect=True), mimetype='multipart/x-mixed-replace; boundary=frame')

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=False)

import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import RPi.GPIO as GPIO
import time

class UltrasonicHandler:
    def _init_(self, trigger_pin, echo_pin):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.output(self.trigger_pin, False)
        time.sleep(2)  # Allow sensor to settle

    def get_distance(self):
        # Send trigger pulse
        GPIO.output(self.trigger_pin, True)
        time.sleep(0.00001)
        GPIO.output(self.trigger_pin, False)

        # Wait for echo response
        pulse_start, pulse_end = 0, 0
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = (pulse_duration * 34300) / 2  # Convert time to distance in cm
        return round(distance, 2)

    def cleanup(self):
        GPIO.cleanup()

# To import this module in another file:
# from ultrasonic_handler import UltrasonicHandler
# sensor = UltrasonicHandler(trigger_pin=23, echo_pin=24)
# distance = sensor.get_distance()
# print(f"Measured Distance: {distance} cm")
# sensor.cleanup()

# Example usage
if _name_ == "_main_":
    sensor = UltrasonicHandler(trigger_pin=20, echo_pin=21)
    try:
        while True:
            distance = sensor.get_distance()
            print(f"Distance: {distance} cm")
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.cleanup() 
