import cv2
import numpy as np
import csv
from datetime import datetime
import os
import time

# ==========================
# FILE PATHS
# ==========================
MODEL_PATH = "trainer.yml"
LABELS_PATH = "labels.npy"
ATTENDANCE_FILE = "attendance.csv"

# ==========================
# TUNING
# ==========================
CONFIDENCE_THRESHOLD = 100   # below this = recognised, above = Unknown
CONFIRM_FRAMES = 3
MIN_FACE_AREA = 4000

MAX_TIME =  2 * 60
MIN_PRESENT =  1 * 60

# ==========================
# LOAD MODELS
# ==========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

label_map = np.load(LABELS_PATH, allow_pickle=True).item()

# ==========================
# TRACKING STRUCTURES
# ==========================
frame_counter = {}
attendance_timer = {}

# ==========================
# CAMERA
# ==========================
cap = cv2.VideoCapture(0)
print("🎥 Recognition running (press q to quit)")
print("📊 Confidence scores printed below — lower = better match")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_eq = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))

    current_time = time.time()

    for (x, y, w, h) in faces:

        if w * h < MIN_FACE_AREA:
            continue

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))

        label, confidence = recognizer.predict(face)

        # DEBUG: see actual confidence values in terminal
        name_guess = label_map.get(label, "Unknown")
        print(f"  → Label: {name_guess} | Confidence: {confidence:.1f}")

        # LBPH: 0 = perfect match, higher = worse.
        # If confidence > threshold OR label not in map → Unknown
        if confidence <= CONFIDENCE_THRESHOLD and label in label_map:
            name = label_map[label]
            frame_counter[name] = frame_counter.get(name, 0) + 1

            if frame_counter[name] >= CONFIRM_FRAMES:

                if name not in attendance_timer:
                    attendance_timer[name] = {
                        "last_seen": current_time,
                        "total_time": 0,
                        "present": False
                    }
                else:
                    elapsed = current_time - attendance_timer[name]["last_seen"]
                    attendance_timer[name]["total_time"] += elapsed
                    attendance_timer[name]["last_seen"] = current_time

                if attendance_timer[name]["total_time"] >= MIN_PRESENT:
                    attendance_timer[name]["present"] = True

                minutes = int(attendance_timer[name]["total_time"] / 60)
                text = f"{name} ({minutes} min)"
                color = (0, 255, 0)
            else:
                text = "Verifying..."
                color = (0, 255, 255)

        else:
            # Unregistered person or poor match → always Unknown
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("LBPH Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ==========================
# FINAL CSV WRITE
# ==========================
today = datetime.now().strftime("%Y-%m-%d")

with open(ATTENDANCE_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Date", "Name", "Status", "Time (minutes)"])

    all_students = set(label_map.values())

    for student in all_students:
        if student in attendance_timer and attendance_timer[student]["present"]:
            mins = int(attendance_timer[student]["total_time"] / 60)
            writer.writerow([today, student, "Present", mins])
        else:
            writer.writerow([today, student, "Absent", 0])

print("✅ Attendance finalized and saved")