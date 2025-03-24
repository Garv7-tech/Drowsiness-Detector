# Import required libraries
import cv2
import mediapipe as mp
import numpy as np
import pygame
from twilio.rest import Client
from config import *
import time
import csv
import datetime

# Initialize mixer for alarm sound
pygame.mixer.init()
pygame.mixer.music.load(ALARM_PATH)

# Initialize face mesh detection
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# Blink rate variables
blink_count = 0
start_time = time.time()

# Create the log file with headers if it doesn't exist
log_file_path = "drowsiness_log.csv"
try:
    with open(log_file_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Timestamp", "Drowsy", "Blink Rate", "Pitch", "Yaw", "Roll"])
except Exception as e:
    print(f"Error creating log file: {e}")

# Function to play alarm sound
def play_alarm():
    try:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound: {e}")

# Function to send SMS alert using Twilio
def send_alert():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="Drowsiness detected! Please take a break.",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print(f"Alert sent: {message.sid}")
    except Exception as e:
        print(f"Error sending alert: {e}")

# Function to log drowsiness detection events
def log_drowsiness_event(drowsy, blink_rate, pitch, yaw, roll):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file_path, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([timestamp, drowsy, blink_rate, pitch, yaw, roll])
    except Exception as e:
        print(f"Error writing to log file: {e}")

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Mouth Aspect Ratio (MAR) calculation
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(np.array(mouth[2]) - np.array(mouth[6]))
    B = np.linalg.norm(np.array(mouth[3]) - np.array(mouth[5]))
    C = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[4]))
    mar = (A + B) / (2.0 * C)
    return mar

# Detect head pose using solvePnP
def head_pose_estimation(landmarks):
    def get_coords(idx):
        return (landmarks.landmark[idx].x * 640, landmarks.landmark[idx].y * 480)

    image_points = np.array([
        get_coords(1),    # Nose tip
        get_coords(199),  # Chin
        get_coords(33),   # Left eye left corner
        get_coords(263),  # Right eye right corner
        get_coords(61),   # Left mouth corner
        get_coords(291)   # Right mouth corner
    ], dtype=np.float32)

    model_points = np.array([
        (0.0, 0.0, 0.0),      # Nose tip
        (0.0, -330.0, -65.0), # Chin
        (-225.0, 170.0, -135.0), # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    focal_length = 1.0 * 640
    center = (640 / 2, 480 / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = np.degrees(angles)
    return pitch, yaw, roll

# Drowsiness detection
# def detect_drowsiness(face_landmarks):
#     global blink_count, start_time
#     drowsy = False

#     left_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [33, 160, 158, 133, 153, 144]]
#     right_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [362, 385, 387, 263, 373, 380]]
#     mouth = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [61, 62, 63, 64, 65, 66, 67]]

#     ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
#     mar = mouth_aspect_ratio(mouth)
#     pitch, yaw, roll = head_pose_estimation(face_landmarks)

#     if ear < EAR_THRESHOLD:
#         blink_count += 1
#         elapsed_time = time.time() - start_time
#         blink_rate = blink_count / elapsed_time * 60
#     else:
#         blink_rate = 0

#     if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD or abs(pitch) > HEAD_POSE_THRESHOLD:
#         drowsy = True

#     return drowsy, blink_rate, pitch, yaw, roll


# Drowsiness detection
def detect_drowsiness(face_landmarks):
    global blink_count, start_time
    drowsy = False

    # Eye landmarks
    left_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [33, 160, 158, 133, 153, 144]]
    right_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [362, 385, 387, 263, 373, 380]]
    mouth = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [61, 62, 63, 64, 65, 66, 67]]

    # Calculate EAR and MAR
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    mar = mouth_aspect_ratio(mouth)

    # Head pose estimation
    pitch, yaw, roll = head_pose_estimation(face_landmarks)

    # Blink rate calculation
    if ear < EAR_THRESHOLD:
        blink_count += 1
        elapsed_time = time.time() - start_time
        blink_rate = blink_count / elapsed_time * 60
    else:
        blink_rate = 0

    # Detect drowsiness based on EAR, MAR, and head pose
    if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD or abs(pitch) > HEAD_POSE_THRESHOLD:
        drowsy = True

    return drowsy, blink_rate, pitch, yaw, roll


# Drowsiness detection variables
drowsiness_start = None
DROWSINESS_DURATION = 10  # Minimum duration to detect drowsiness (in seconds)

# Main detection loop
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                drowsy, blink_rate, pitch, yaw, roll = detect_drowsiness(face_landmarks)

                # Handle sustained drowsiness detection
                if drowsy:
                    if drowsiness_start is None:
                        drowsiness_start = time.time()
                    elapsed_drowsiness = time.time() - drowsiness_start
                    if elapsed_drowsiness >= DROWSINESS_DURATION:
                        print("Drowsiness detected!")
                        play_alarm()
                        send_alert()
                        log_drowsiness_event(drowsy, blink_rate, pitch, yaw, roll)
                        drowsiness_start = None  # Reset after sending the alert
                else:
                    drowsiness_start = None  # Reset the drowsiness start time

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Main detection loop
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                drowsy, blink_rate, pitch, yaw, roll = detect_drowsiness(face_landmarks)

                if drowsy:
                    print("Drowsiness detected!")
                    play_alarm()
                    send_alert()

                log_drowsiness_event(drowsy, blink_rate, pitch, yaw, roll)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
