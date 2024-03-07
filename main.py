from ultralytics import YOLO
import cv2
import csv
import math
import time
import os
from conditions import Weather, Location
import logging
from flask import Flask, Response, render_template

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
LOCATION = Location.INDOOR

# Weather conditions. None if indoors. Manually set this value based on the conditions.
WEATHER = None if LOCATION == Location.INDOOR else Weather.SUNNY

# Classes that YOLO model is limited to detect.
class_names = ["person"]

# A boolean to stop the recording function to not record when already recording.
recording = True

# Flask related
# frame = None
running = True
app = Flask(__name__)


@app.route('/')
def video():
    return render_template('video.html')


@app.route('/video_feed')
def video_feed():
    return Response(stream_detect_people(), mimetype='multipart/x-mixed-replace; boundary=frame')


def stream_detect_people():
    """
    This function generates frames from the camera and yields them for streaming.
    """
    global MINUTES, RUNTIME

    detected_people = {}
    confidences = []
    output_path = "person_clips"
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.INFO)

    average_luminance = measure_luminance()
    print("Average luminance:", average_luminance)

    model = YOLO(MODEL_PATH)
    cap = initialize_video_capture()
    logging.info("Initialized video capture")
    start_time = time.time()

    while time.time() - start_time < RUNTIME:
        success, img = cap.read()
        if not success:
            logging.error("Error reading frame from camera")
            break

        results = model(img, stream=True)
        update_detected_people(results, detected_people, confidences)

        if detected_people:
            save_img(img, cv2, output_path)

        if cv2.waitKey(1) == ord('q'):
            logging.warning("cv2 stopped (pressed 'q')")
            break

        draw_detections_and_info(img, detected_people)

        ret, buffer = cv2.imencode('.jpg', img)
        bytes_frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n')

        detected_people.clear()
        confidences.clear()

    logging.info(f"{MINUTES} minutes is over. Detection stopped.")
    confidences = remove_zeros(confidences)
    average_accuracy = sum(confidences) / len(confidences) if confidences else 0

    print("Average accuracy:", average_accuracy)
    save_results(average_luminance, average_accuracy, "results.csv")

    cap.release()
    cv2.destroyAllWindows()


def stop_server():
    global running
    running = False


@app.route('/stop')
def stop():
    stop_server()
    return "Server stopped."


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


def draw_detections_and_info(img, detected_people):
    """
    draws bounding boxes and confidence information for detected people on the image.
    :param img: the image to draw on.
    :param detected_people: dictionary storing detected people data (ID, confidence, bounding box).

    """
    for _, (confidence, (x1, y1, x2, y2)) in detected_people.items():
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(img, f"Person - Conf: {confidence:.2f}", (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if detected_people:
        print(f"Number of people detected: {len(detected_people)}")


# Function to save results to Excel file
def save_results(luminance, average_accuracy, output_file):
    """
    This function saves the average luminance, weather, location and accuracy to a CSV file,
    :param luminance: The average luminance value.
    :param average_accuracy: The average accuracy
    :param output_file: Name of the output csv file
    """
    global WEATHER, LOCATION
    # Check if the file exists
    if not os.path.exists(output_file):
        # Create the file and add headers
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Average Luminance", "Weather", "Location", "Average Accuracy"])

    # Append data to existing file
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([luminance, "" if WEATHER is None else WEATHER.value, LOCATION.value, average_accuracy])

    logging.info("Instance/data sample added to csv file.")


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
    filename = f"{output_path}/person{int(time.time())}.jpg"
    cv.imwrite(filename, img)
    logging.info(f"Image saved: {filename}")


if __name__ == "__main__":
    # Start the Flask app

    app.run(host='0.0.0.0', port=80, debug=True)
    # main()
