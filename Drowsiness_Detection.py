from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import threading
import tkinter as tk
from tkinter import Button, Label
import os

# Initialize sound
mixer.init()
mixer.music.load("alert.mp3.wav")

# Constants
thresh = 0.25
frame_check = 25

# Dlib face detector and predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Global variables
cap = None
running = False
flag = 0

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def start_camera():
    """Start the camera in a separate thread"""
    global cap, running, flag
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        flag = 0
        threading.Thread(target=run_camera, daemon=True).start()

def stop_camera():
    """Stop the camera"""
    global cap, running
    running = False
    if cap:
        cap.release()
        cap = None

def exit_program():
    """Stop the camera and forcefully exit the program"""
    stop_camera()
    os._exit(0)  # Forcefully exit all threads and GUI

def run_camera():
    """Run the camera loop"""
    global cap, running, flag

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=650)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_camera()
    cv2.destroyAllWindows()

# GUI window
root = tk.Tk()
root.title("Drowsiness Detection")
root.geometry("400x300")

# Labels and buttons
label = Label(root, text="Drowsiness Detection", font=("Helvetica", 16))
label.pack(pady=20)

start_button = Button(root, text="Start Camera", command=start_camera, bg="green", fg="white", font=("Helvetica", 14))
start_button.pack(pady=10)

stop_button = Button(root, text="Stop Camera", command=stop_camera, bg="red", fg="white", font=("Helvetica", 14))
stop_button.pack(pady=10)

exit_button = Button(root, text="Exit", command=exit_program, bg="gray", fg="white", font=("Helvetica", 14))
exit_button.pack(pady=10)

# Start GUI event loop
root.mainloop()
