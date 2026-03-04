"""
Module 3 — Face Recognition & Attendance Tracking
===================================================
Loads the trained LBPH model, recognises students via webcam,
tracks how long each student is on screen, and writes results
to  attendance.csv  when you quit.

Usage:
    python recognize.py

Controls:
    Q  — quit and save attendance
"""

import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CAMERA_INDEX         = 0      # built-in webcam

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "..", "trainer.yml")
LABELS_PATH     = os.path.join(os.path.dirname(__file__), "..", "labels.npy")
ATTENDANCE_FILE = os.path.join(os.path.dirname(__file__), "..", "attendance.csv")

# Recognition tuning
# LBPH distance: 0 = perfect match, higher = worse.
# Tune CONFIDENCE_THRESHOLD based on the scores printed in the terminal:
#   score < 50  → very confident match
#   50–80       → acceptable match  (threshold lives here)
#   80–100      → weak match
#   > 100       → almost certainly wrong person
CONFIDENCE_THRESHOLD = 90
CONFIRM_FRAMES       = 8     # consecutive confirmed frames before starting timer
MIN_FACE_AREA        = 4000  # ignore tiny detections (px²)

# Attendance rules
MIN_PRESENT_SECONDS  = 20    # student must appear for at least this long → "Present"

# ──────────────────────────────────────────────
# LOAD MODEL + LABELS
# ──────────────────────────────────────────────
model_path  = os.path.abspath(MODEL_PATH)
labels_path = os.path.abspath(LABELS_PATH)

if not os.path.exists(model_path):
    print(f"❌  Model not found: {model_path}")
    print("    Run  train.py  first.")
    exit(1)

if not os.path.exists(labels_path):
    print(f"❌  Labels file not found: {labels_path}")
    print("    Run  train.py  first.")
    exit(1)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

label_map: dict = np.load(labels_path, allow_pickle=True).item()
print(f"✅  Model loaded — {len(label_map)} registered student(s): {list(label_map.values())}")

# ──────────────────────────────────────────────
# TRACKING STATE
# ──────────────────────────────────────────────
# frame_counter  — frames a name has been consecutively confirmed
# attendance     — per-student timing dict
#     "last_seen"  : float timestamp of the last frame they were seen
#     "total_time" : float total seconds accumulated
#     "present"    : bool — True once >= MIN_PRESENT_SECONDS

frame_counter: dict = {}
attendance:   dict = {}

# ──────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌  Cannot open camera {CAMERA_INDEX}.")
    exit(1)

print("🎥  Recognition running — press  Q  to quit and save attendance.\n")

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
seen_this_frame: set = set()   # which names were spotted each frame

# Last-known face buffer — keeps box visible during brief detection gaps
last_face   = None   # (x, y, w, h) from last successful detection
last_label  = ""     # name shown in that position
last_color  = (200, 200, 200)
miss_streak = 0      # frames since last successful detection
MAX_MISS    = 8      # frames to hold last position before clearing

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Frame grab failed.")
        break

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)   # improve contrast for recognition

    faces = face_cascade.detectMultiScale(
        gray_eq, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60)
    )

    now = time.time()
    seen_this_frame.clear()

    for (x, y, w, h) in faces:
        if w * h < MIN_FACE_AREA:
            continue   # skip tiny / distant faces

        # Apply equalizeHist to the crop — MUST match what train.py produces
        face_roi = cv2.equalizeHist(cv2.resize(gray[y: y + h, x: x + w], (100, 100)))
        label, confidence = recognizer.predict(face_roi)

        # DEBUG: prints live — use these numbers to fine-tune CONFIDENCE_THRESHOLD
        name_guess = label_map.get(label, "?")
        print(f"  score={confidence:.1f}  guess={name_guess}  threshold={CONFIDENCE_THRESHOLD}")

        if confidence <= CONFIDENCE_THRESHOLD and label in label_map:
            name = label_map[label]
            frame_counter[name] = frame_counter.get(name, 0) + 1

            if frame_counter[name] >= CONFIRM_FRAMES:
                seen_this_frame.add(name)

                if name not in attendance:
                    attendance[name] = {
                        "last_seen":  now,
                        "total_time": 0.0,
                        "present":    False,
                    }
                else:
                    elapsed = now - attendance[name]["last_seen"]
                    if elapsed < 10:
                        attendance[name]["total_time"] += elapsed
                    attendance[name]["last_seen"] = now

                if attendance[name]["total_time"] >= MIN_PRESENT_SECONDS:
                    attendance[name]["present"] = True

                total_mins = int(attendance[name]["total_time"] / 60)
                total_secs = int(attendance[name]["total_time"] % 60)
                label_text  = f"{name}  {total_mins}m {total_secs:02d}s"
                box_color   = (0, 255, 0)   # green = recognised
            else:
                label_text = "Verifying..."
                box_color  = (0, 200, 255)  # yellow = building confidence

        else:
            # Unknown or poor match
            frame_counter.pop(label_map.get(label, ""), None)  # reset counter
            label_text = "Unknown"
            box_color  = (0, 0, 255)   # red

        # Update last-known buffer on every successful detection
        last_face  = (x, y, w, h)
        last_label = label_text
        last_color = box_color
        miss_streak = 0

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # ── If no face detected this frame, hold the last known box briefly ──
    if len(faces) == 0 and last_face is not None:
        miss_streak += 1
        if miss_streak <= MAX_MISS:
            lx, ly, lw, lh = last_face
            # Draw faded (half-alpha look via lighter colour)
            faded_color = tuple(max(0, c - 80) for c in last_color)
            cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), faded_color, 1)
            cv2.putText(frame, last_label, (lx, ly - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, faded_color, 1)
        else:
            last_face = None   # too many misses — clear buffer

    # ── HUD — list confirmed students and their time ──
    hud_y = 25
    for sname, data in attendance.items():
        mins  = int(data["total_time"] / 60)
        secs  = int(data["total_time"] % 60)
        mark  = "✓" if data["present"] else "…"
        hud   = f"{mark} {sname}: {mins}m {secs:02d}s"
        color = (0, 255, 120) if data["present"] else (200, 200, 200)
        cv2.putText(frame, hud, (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        hud_y += 25

    cv2.imshow("AutoAttender — Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ──────────────────────────────────────────────
# WRITE ATTENDANCE CSV
# ──────────────────────────────────────────────
today      = datetime.now().strftime("%Y-%m-%d")
time_stamp = datetime.now().strftime("%H:%M:%S")
att_path   = os.path.abspath(ATTENDANCE_FILE)

# Check which records for today already exist so we don't double-write
existing_today: set = set()
if os.path.exists(att_path):
    with open(att_path, newline="") as rf:
        for row in csv.reader(rf):
            if len(row) >= 2 and row[0] == today:
                existing_today.add(row[1])

all_students = set(label_map.values())

with open(att_path, "a", newline="") as f:
    writer = csv.writer(f)

    # Write header only if file is new / empty
    if os.path.getsize(att_path) == 0:
        writer.writerow(["Date", "Name", "Status", "Minutes", "Recorded At"])

    for student in sorted(all_students):
        if student in existing_today:
            continue  # already recorded for today — skip

        if student in attendance and attendance[student]["present"]:
            mins   = int(attendance[student]["total_time"] / 60)
            status = "Present"
        else:
            mins   = int(attendance.get(student, {}).get("total_time", 0) / 60)
            status = "Absent"

        writer.writerow([today, student, status, mins, time_stamp])

print("\n📊  Session Summary")
print("─" * 40)
for student in sorted(all_students):
    if student in attendance:
        mins  = int(attendance[student]["total_time"] / 60)
        secs  = int(attendance[student]["total_time"] % 60)
        mark  = "✅ Present" if attendance[student]["present"] else "❌ Absent"
        print(f"  {mark}  {student}  —  {mins}m {secs:02d}s on screen")
    else:
        print(f"  ❌ Absent   {student}  —  never appeared")

print("─" * 40)
print(f"✅  Attendance saved → {att_path}")
