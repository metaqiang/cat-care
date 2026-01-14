from flask import Flask, render_template, Response, jsonify, request
import cv2
import Adafruit_DHT
from flasgger import Swagger
import threading
import time
from datetime import datetime
from smbus2 import SMBus
from sgp30 import Sgp30
import RPi.GPIO as GPIO

app = Flask(__name__)
swagger_config = Swagger.DEFAULT_CONFIG

Swagger(
    app,
    config=swagger_config,
    template={
        "swagger": "2.0",
        "uiversion": 3,
    },
)

DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 4

# --- MG90S Servo Motor ---
SERVO_PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Start PWM with 50Hz frequency
servo_pwm = GPIO.PWM(SERVO_PIN, 50)
servo_pwm.start(0)


def set_servo_angle_hw(angle):
    """Control servo angle using PWM"""
    duty_cycle = (angle / 18) + 2.5  # Convert angle to duty cycle
    servo_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.8)  # Wait for servo to reach and stabilize at target position
    servo_pwm.ChangeDutyCycle(0)  # Stop PWM signal to prevent jitter


# --- SGP30 Air Quality Sensor ---
class AirQualitySensor:
    def __init__(self):
        self.co2 = None
        self.tvoc = None
        self.bus = None
        self.sgp = None

        self.keep_running = True
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.daemon = True
        self.thread.start()

    def _read_loop(self):
        """Background thread: continuously read CO2 and TVOC from SGP30"""
        try:
            self.bus = SMBus(1)
            self.sgp = Sgp30(self.bus, baseline_filename="/tmp/mySGP30_baseline")
            self.sgp.init_sgp()
            time.sleep(10)  # Warm-up time

            while self.keep_running:
                try:
                    result = self.sgp.read_measurements()
                    if result:
                        self.co2 = result.data[0]
                        self.tvoc = result.data[1]
                except Exception:
                    pass  # Silently handle read failures, keep previous values

                time.sleep(1)
        except Exception:
            pass  # Silently handle init failures, co2/tvoc remain None

    def get_readings(self):
        """Get current CO2 and TVOC readings"""
        return self.co2, self.tvoc

    def __del__(self):
        self.keep_running = False
        if self.bus:
            try:
                self.bus.close()
            except Exception:
                pass


# --- Optimization 1: global singleton camera class ---
class VideoCamera(object):
    def __init__(self):
        # 0 refers to the first camera device
        self.video = cv2.VideoCapture(0)

        # --- Optimization 2: lower resolution ---
        # 320x240 is sufficient for remote monitoring and reduces bandwidth to about 1/4
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Try to set hardware FPS; many cameras don't support it, so we control the rate in the thread
        self.video.set(cv2.CAP_PROP_FPS, 15)

        # Reduce camera buffer to minimize latency (get freshest frame)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.frame_id = 0  # Frame counter for tracking new frames

        # Use Condition for efficient new-frame notification
        self.condition = threading.Condition()

        # Start a background thread to read frames to avoid blocking web responses
        self.keep_running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

    def _capture_loop(self):
        """Background thread: responsible for reading the latest frames from hardware"""
        while self.keep_running:
            success, image = self.video.read()
            if success:
                # Add timestamp overlay on the frame
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Draw shadow first (offset by 1 pixel) for better visibility
                cv2.putText(
                    image,
                    timestamp,
                    (11, 26),  # Shadow position: slightly offset
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # Font scale
                    (0, 0, 0),  # Black shadow
                    2,  # Thickness
                    cv2.LINE_AA,
                )
                # Draw main text on top
                cv2.putText(
                    image,
                    timestamp,
                    (10, 25),  # Position: top-left corner
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White text
                    1,
                    cv2.LINE_AA,
                )

                # --- Optimization 3: JPEG compression quality ---
                # Quality set to 70 as a trade-off between image quality and bandwidth
                # (can be lowered to 40-50 to reduce bandwidth further)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                ret, jpeg = cv2.imencode(".jpg", image, encode_param)

                if ret:
                    with self.condition:
                        self.frame = jpeg.tobytes()
                        self.frame_id += 1
                        # Notify all waiting clients that a new frame is available
                        self.condition.notify_all()

            # --- Optimization 4: physical rate limiting ---
            # Sleep to limit FPS to ~12-15 fps, saving CPU and bandwidth
            time.sleep(0.07)

    def get_frame_blocking(self, last_frame_id=0, timeout=1.0):
        """
        Wait for a new frame and return it.
        Returns (frame_bytes, frame_id) or (None, last_frame_id) on timeout.
        This ensures clients always get the latest frame without polling.
        """
        with self.condition:
            # Wait until a newer frame is available or timeout
            if self.frame_id <= last_frame_id:
                self.condition.wait(timeout=timeout)

            # Return current frame and its ID
            return self.frame, self.frame_id

    def get_frame(self):
        """Get the current latest frame (non-blocking, for compatibility)"""
        with self.condition:
            return self.frame

    def __del__(self):
        self.keep_running = False
        if self.video.isOpened():
            self.video.release()


# Initialize global camera object (singleton)
# This ensures only one camera connection is opened regardless of concurrent users
global_camera = VideoCamera()

# Initialize global air quality sensor (singleton)
global_air_sensor = AirQualitySensor()


def gen(camera):
    """
    Generator that yields video frames for MJPEG streaming.
    Uses blocking wait to ensure each client always receives the latest frame.
    """
    last_frame_id = 0

    while True:
        # Block until a new frame is available (no polling, no duplicate frames)
        frame, frame_id = camera.get_frame_blocking(last_frame_id, timeout=1.0)

        if frame is not None and frame_id > last_frame_id:
            last_frame_id = frame_id
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
            )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # Use the global `global_camera` instead of creating a new VideoCamera each time
    return Response(
        gen(global_camera), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/sensor_readings", methods=["GET"])
def get_sensor_readings():
    """Get sensor readings
    ---
    responses:
      200:
        description: Sensor readings
    """
    humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
    co2, tvoc = global_air_sensor.get_readings()

    return (
        jsonify(
            {"temperature": temperature, "humidity": humidity, "co2": co2, "tvoc": tvoc}
        ),
        200,
    )


@app.route("/feed", methods=["GET"])
def feed():
    """Trigger feeding action
    ---
    responses:
      200:
        description: Feeding action completed successfully
    """
    set_servo_angle_hw(150)
    set_servo_angle_hw(180)
    return jsonify({"status": "success", "message": "Feeding completed"}), 200


@app.route("/servo", methods=["POST"])
def set_servo_angle():
    """Set servo angle
    ---
    parameters:
      - name: angle
        in: body
        schema:
          type: object
          properties:
            angle:
              type: number
    responses:
      200:
        description: Servo angle set successfully
    """
    data = request.get_json()
    angle = data.get("angle")

    if angle is None:
        return jsonify({"error": "angle is required"}), 400

    if not (0 <= angle <= 180):
        return jsonify({"error": "angle must be between 0 and 180"}), 400

    set_servo_angle(angle)
    return jsonify({"angle": angle}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
