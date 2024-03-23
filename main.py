from pathlib import Path
from ultralytics import YOLO
import cv2
import csv
import math
import time
import os
import log
from log import web_logs
from conditions import Weather, Location
from flask import Flask, request, redirect, url_for, render_template, make_response, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

# import serial # (We use RPI.GPIO)

# -------------Configuration--------------------
# Path to the model (e.g., yolov8n.pt, yolov8x.pt)
MODEL_PATH = "models/yolov8n.pt"

# How long should we detect people for? (in minutes)
MINUTES = 10

# Converted to seconds to use with the 'time' library
RUNTIME = MINUTES * 60

# Threshold for considering bounding boxes as overlapping, we have to fine-tune this (P.S 0.5 doesn't work)
OVERLAP_THRESHOLD = 1

# How many times should we poll the luminance sensor for to get an average reading?
LUMINANCE_RECORDINGS = 5

# Camera location. Manually set this value.
# Weather conditions. None if indoors. Manually set this value based on the conditions.
location, weather = None, None

# Classes that YOLO model is limited to detect.
class_names = ["person"]

# Number of JPEG files already present in the OUTPUT_PATH
initial_images_in_person_clips = 0

# Directory where images are stored.
OUTPUT_PATH = "static/person_clips"

# A flag indicating whether to clear the images in the OUTPUT_PATH at the start of each session (video streaming).
INITIALLY_CLEAR_IMAGES = True

# -------------Flask related--------------------

# A flag that controls the main loop of the video processing and detection functionality.
# When running is True, the application will continuously capture frames from the camera,
# detect people using the YOLO model, and save frames containing detected people.
# When running is False, the application will stop capturing frames and processing the video stream.
running = False

# Define a Flask app name
app = Flask(__name__)

# Stores a secret key used for cryptographic purposes within the Flask application.
app.config['SECRET_KEY'] = secrets.token_urlsafe(32)

# Holds an instance of the Flask-Login extension's LoginManager class.
login_manager = LoginManager()
login_manager.init_app(app)


class User(UserMixin):
    def __init__(self, username, password):
        self.username = username
        self.password = password

    def get_id(self):
        """
        Returns a unique identifier for the user.
        :return: Username of the user.
        """
        return self.username


@app.route('/set_location', methods=['POST'])
@login_required
def set_location():
    global location

    # Extract the location from the request body
    data = request.get_json()
    if data and 'location' in data:
        selected_location = data['location']
        try:
            # Convert the string value to the corresponding Location enum member
            location = Location(selected_location)
            return "Location updated successfully."
        except ValueError:
            return "Invalid location selection.", 400  # Bad request

    return "Error updating location.", 400  # Bad request


@app.route('/set_weather', methods=['POST'])
@login_required
def set_weather():
    global weather

    # Extract the location from the request body
    data = request.get_json()
    if data and 'weather' in data:
        selected_weather = data['weather']
        try:
            # Convert the string value to the corresponding Location enum member
            weather = Weather(selected_weather)
            return "Weather updated successfully."
        except ValueError:
            return "Invalid weather selection.", 400  # Bad request

    return "Error updating weather.", 400  # Bad request


@login_manager.user_loader
def load_user(username):
    """
    Function to load the user object by username (Flask-Login requirement)
    :param username: The entered username.
    :return: The only user
    """
    global user
    return user if username == user.username else None


@app.route('/', methods=['GET', 'POST'])
def login():
    """
    This function handles login requests for the application.

    It checks for the HTTP method used (GET or POST) and performs the following actions:

    - GET: Renders the login form template (login.html).
    - POST:
        1. Extracts username and password from the form submission.
        2. Validates user credentials:
            - Checks if a user is already stored in the global variable `user`.
            - Compares the submitted username with the stored user's username.
            - Verifies the password using the `check_password_hash` function.
        3. If credentials are valid:
            - Calls `login_user` function.
            - Redirects the user to the main page (`video` route).
        4. If credentials are invalid:
            - Returns an error message indicating invalid username or password.
    """
    global user

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Validate user credentials
        if user and username == user.username and check_password_hash(user.password, password):
            login_user(user)
            log.success('Successfully logged in')
            return redirect(url_for('video'))  # Redirect to the main page after successful login
        return 'Invalid username or password'  # Error message on failed login
    return render_template('login.html')  # Render login form


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/video')
@login_required
def video():
    return render_template('video.html')


@app.route('/video_feed')
@login_required
def video_feed():
    return Response(stream_detect_people(), mimetype='multipart/x-mixed-replace; boundary=frame')


def stream_detect_people():
    """
    Streams video frames from the camera, detects people, and saves detections.
    :except RuntimeError: If an error occurs while reading a frame from the camera.
    """

    # --- Initialization ---
    global MINUTES, RUNTIME, OUTPUT_PATH, INITIALLY_CLEAR_IMAGES, initial_images_in_person_clips

    detected_people = {}
    confidences = []

    if INITIALLY_CLEAR_IMAGES:
        clear_images()
    else:
        initial_images_in_person_clips = count_jpgs()

    # Measure average luminance from the sensor.
    average_luminance = measure_luminance()
    log.success(f"Average luminance: {average_luminance}")

    model = YOLO(MODEL_PATH)
    cap = initialize_video_capture()
    log.info("Initialized video capture")

    while not running:
        continue

    log.info("Started to stream and detect")
    start_time = time.time()

    # --- Detection and streaming loop ---
    while time.time() - start_time < RUNTIME and running:
        # Read a frame from the camera
        success, img = cap.read()

        if not success:
            log.error("Error reading frame from camera")
            break

        # Run object detection on the frame
        results = model(img, stream=True)

        # Update detected people and their confidence scores
        update_detected_people(results, detected_people, confidences)

        # Save frames with detected people
        if detected_people:
            save_img(img, cv2, OUTPUT_PATH)

        if cv2.waitKey(1) == ord('q'):
            log.error("cv2 stopped (pressed 'q')")
            break

        # Draw detections and information on the frame
        draw_detections_and_info(img, detected_people)

        # Encode the frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', img)
        bytes_frame = buffer.tobytes()

        # Yield the encoded frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n')

        # Clear the detected people for the next frame
        detected_people.clear()
        # confidences.clear()

    # --- Wrap-up ---
    log.info(f"{MINUTES} minutes is over or manually stopped")

    # Calculate average accuracy (excluding zero-confidence detections)
    confidences = remove_zeros(confidences)
    average_accuracy = sum(confidences) / len(confidences) if confidences else 0

    log.success(f"Average accuracy: {average_accuracy}")

    # Save results (luminance and accuracy) to a CSV file
    save_results(average_luminance, average_accuracy, "results.csv")

    # Release video capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()
    log.error("OpenCV destroyed all windows (ignore if not opened)")


@app.route('/stop')
@login_required
def stop():
    global running, OUTPUT_PATH
    running = False
    log.success(f"Successfully saved {abs(initial_images_in_person_clips - count_jpgs())} images!")

    return "Server stopped."


@app.route('/images')
@login_required
def get_images():
    """
    This function retrieves the image paths from the output directory and returns them as a JSON response.
    """
    global OUTPUT_PATH
    image_paths = [os.path.join(OUTPUT_PATH, filename) for filename in os.listdir(OUTPUT_PATH) if
                   filename.endswith(".jpg")]
    return jsonify(image_paths)


@app.route('/start')
@login_required
def start():
    global running, location, weather
    log.success(f"Location set to {location.value}")
    log.success(f"Weather set to {weather.value}")
    running = True
    return "Server started."


@app.route('/logs')
@login_required
def get_logs():
    logs = '\n'.join(list(web_logs))  # Join logs using newline character
    response = make_response(logs)
    response.headers['Content-Type'] = 'text/html'
    web_logs.clear()
    return response


def remove_zeros(data):
    """
    Only retains non-zeros values given a list.
    :param data: A list of confidences
    :return: A list of only non-zero values.
    """
    return [x for x in data if x != 0]


def measure_luminance():
    """
    Function to measure average luminance using the Arduino photo-resistor
    :return:
    """
    # ser = serial.Serial('/dev/ttyACM0', 9600)  # Adjust port and baud rate if needed
    total_luminance = 0
    for _ in range(LUMINANCE_RECORDINGS):
        # reading = float(ser.readline().decode())  # Replace with your sensor reading logic
        # total_luminance += reading
        pass
    return total_luminance / LUMINANCE_RECORDINGS


def initialize_video_capture():
    """
    this function initializes video capture from the Raspberry Pi camera.
    it sets the resolution to 640x480 pixels.
    :return: cap
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height
    return cap


def draw_detections_and_info(img, detected_people, display_detections=False):
    """
    draws bounding boxes and confidence information for detected people on the image.
    :param display_detections: prints number of detected people if True.
    :param img: the image to draw on.
    :param detected_people: dictionary storing detected people data (ID, confidence, bounding box).

    """
    for _, (confidence, (x1, y1, x2, y2)) in detected_people.items():
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(img, f"Person - Conf: {confidence:.2f}", (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if detected_people and display_detections:
        log.success(f"Number of people detected: {len(detected_people)}")


# Function to save results to Excel file
def save_results(luminance, average_accuracy, output_file):
    """
    This function saves the average luminance, weather, location and accuracy to a CSV file,
    :param luminance: The average luminance value.
    :param average_accuracy: The average accuracy
    :param output_file: Name of the output csv file
    """
    global weather, location
    # Check if the file exists
    if not os.path.exists(output_file):
        # Create the file and add headers
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Average Luminance", "Weather", "Location", "Average Accuracy"])

    # Append data to existing file
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([luminance, "" if weather is None else weather.value, location.value, average_accuracy])

    log.success("Instance/data sample added to csv file.")


# Function to update detected people data with unique detections and calculate confidence
def update_detected_people(results, detected_people, confidences):
    """
    updates detected_people dictionary with unique detections and their confidence.
    :param results: List of detection results from the model.
    :param detected_people: A list storing detected people data (ID, confidence, bounding box).
    :param confidences: List to store confidence values of unique detections.
    """
    global OVERLAP_THRESHOLD

    for r in results:
        for box in r.boxes:
            if _is_person_detection(box):
                x1, y1, x2, y2 = coordinates_in_int(box)

                is_unique = _check_unique_detection(detected_people, x1, y1, x2, y2)

                if is_unique:
                    confidence = _calculate_confidence(box)
                    # Append all unique confidence values
                    confidences.append(confidence)
                    detected_people[box.id] = (confidence, (x1, y1, x2, y2))


def coordinates_in_int(box):
    """
    Converted the coordinates to integers
    :param box: Bounding box
    :return: x, y coordinates in integers.
    """
    x1, y1, x2, y2 = box.xyxy[0]
    return int(x1), int(y1), int(x2), int(y2)


def _calculate_confidence(box):
    """
    Calculates and rounds the confidence value from the detection based on YOLO's output format.

    Mean Average Precision (mAP): This is a widely used metric that measures both precision and recall across all
    detected classes in our dataset.
    However, mAP might not be the most informative metric for our specific use case as it considers all classes,
    whereas we're only interested in "person" detection.

    Average Precision (AP): Similar to mAP but calculated for a specific class (e.g., "person").
    This provides a more focused evaluation of YOLO's effectiveness in detecting people under various conditions.

    Therefore, we use confidence/accuracy as it is the same as average precision when detecting only one class.

    :param box: Bounding box
    :return: confidence/accuracy
    """
    confidence = math.ceil((box.conf[0] * 100)) / 100  # Multiply by IoU
    return min(confidence, 1.0)  # Ensure value is between 0 and 1


def _check_unique_detection(detected_people, x1, y1, x2, y2):
    """
    checks if the bounding box overlaps with previously detected people beyond a threshold.

    :param detected_people: list storing detected people data (ID, confidence, bounding box).
    :param x1: top-left x coordinate of the new bounding box.
    :param y1: top-left y coordinate of the new bounding box.
    :param x2: bottom-right x coordinate of the new bounding box.
    :param y2: bottom-right y coordinate of the new bounding box.
    :return: true if the detection is unique, False otherwise.
    """
    global OVERLAP_THRESHOLD
    is_unique = True

    for _, prev_box in detected_people.items():
        prev_x1, prev_y1, prev_x2, prev_y2 = prev_box[1]
        intersection_area = max(0, min(x2, prev_x2) - max(x1, prev_x1)) * max(0, min(y2, prev_y2) - max(y1, prev_y1))
        overlap_ratio = intersection_area / min(
            (x2 - x1) * (y2 - y1), (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
        )
        if overlap_ratio > OVERLAP_THRESHOLD:
            is_unique = False
            break
    return is_unique


def _is_person_detection(box):
    """
    Function to check if the detection is for a person
    :param box: bounding box
    :return: True if a person is detection, False otherwise.
    """
    global class_names
    return box.cls is not None and 0 <= box.cls[0] < len(class_names) and class_names[int(box.cls[0])] == "person"


def save_img(img, cv, output_path):
    if not os.path.exists(output_path):
        log.warn(f"{output_path} directory doesn't exist. Creating it...")
        os.makedirs(output_path)  # Create the directory if it doesn't exist

    filename = f"{output_path}/person_{int(time.time())}.jpg"

    try:
        cv.imwrite(filename, img)
    except Exception as e:
        log.error(f"Error saving image: {e}")


def count_jpgs():
    """
    Counts the number of JPG files in a given directory.
    :return: The number of JPG files found in the directory.
    """
    global OUTPUT_PATH
    return len(list(Path(OUTPUT_PATH).rglob("*.jpg")))


def clear_images():
    """
    Removes all JPEGs from static/person_clips directory
    """
    global OUTPUT_PATH

    try:
        os.listdir(OUTPUT_PATH)
    except OSError as e:
        log.error(f"Error accessing output path: {OUTPUT_PATH}. Reason: {e}")
        return  # Exit the function if the path is invalid

    num_files_removed = 0
    for filename in os.listdir(OUTPUT_PATH):
        file_path = os.path.join(OUTPUT_PATH, filename)
        try:
            os.remove(file_path)
            num_files_removed += 1
        except OSError as e:
            log.error(f"Error removing file: {file_path}. Reason: {e}")

    if num_files_removed > 0:
        log.warn(f"Successfully removed {num_files_removed} previous images.")
    else:
        log.warn(f"No images in {OUTPUT_PATH}")


if __name__ == "__main__":
    with open("security/user.txt", "r") as f:
        word1, word2 = f.readline().strip().split(",")
        user = User(str(word1).replace(" ", ""), generate_password_hash(str(word2).replace(" ", "")))

    login_manager.login_view = 'login'

    # Command to generate the files:
    # openssl req -x509 -newkey rsa:4096 -nodes -out flask_cert.pem -keyout flask_key.pem -days 365
    app.run(host ='0.0.0.0', ssl_context=('security/flask_cert.pem', 'security/flask_key.pem'))