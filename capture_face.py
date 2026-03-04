"""
Module 1 — Face Capture
========================
Captures face images for a student from three angles (straight, left, right)
and saves them under  dataset/<student_id>/

Usage:
    python capture_face.py

Controls:
    C  — begin capturing the current angle
    Q  — quit early
"""

import cv2
import os
import time

# ──────────────────────────────────────────────
# CONFIGURATION  (tweak here, not inside loops)
# ──────────────────────────────────────────────
CAMERA_INDEX        = 0           # 0 = built-in webcam
IMAGES_PER_ANGLE    = 60          # photos saved per angle  (3 × 60 = 180 total)
CAPTURE_DELAY_SEC   = 0.1         # min gap between saved frames (~10 fps, good variety)
PREPARE_COUNTDOWN   = 3           # seconds shown before capture starts
FACE_SAVE_SIZE      = (100, 100)  # grayscale face pixel size saved to disk
MIN_BLUR_VARIANCE   = 80.0        # Laplacian variance threshold — skip blurry frames

ANGLES = [
    "Look STRAIGHT at the camera",
    "Tilt face slightly to the LEFT",
    "Tilt face slightly to the RIGHT",
]

# ──────────────────────────────────────────────
# STUDENT INFO
# ──────────────────────────────────────────────
student_id   = input("Enter Student ID   : ").strip()
student_name = input("Enter Student Name : ").strip()

if not student_id or not student_name:
    print("❌  Student ID and Name are both required. Exiting.")
    exit(1)

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", student_id)
os.makedirs(save_dir, exist_ok=True)

# Save name mapping alongside images (one-line text file)
name_file = os.path.join(save_dir, "name.txt")
with open(name_file, "w") as f:
    f.write(student_name)

# ──────────────────────────────────────────────
# LOAD HAAR CASCADE
# ──────────────────────────────────────────────
frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def detect_faces(gray_frame):
    """Try frontal, then profile-left, then profile-right (flipped)."""
    faces = frontal_cascade.detectMultiScale(gray_frame, 1.2, 5, minSize=(60, 60))
    if len(faces):
        return faces
    faces = profile_cascade.detectMultiScale(gray_frame, 1.2, 5, minSize=(60, 60))
    if len(faces):
        return faces
    flipped = cv2.flip(gray_frame, 1)
    return profile_cascade.detectMultiScale(flipped, 1.2, 5, minSize=(60, 60))


def is_sharp(gray_roi, threshold=MIN_BLUR_VARIANCE):
    """Return True when the face crop is sharp enough to save."""
    return cv2.Laplacian(gray_roi, cv2.CV_64F).var() >= threshold


def draw_overlay(frame, line1, line2="", color=(0, 255, 255)):
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 70), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, line1, (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if line2:
        cv2.putText(frame, line2, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# ──────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌  Cannot open camera {CAMERA_INDEX}. Exiting.")
    exit(1)

total_images = len(ANGLES) * IMAGES_PER_ANGLE
print(f"\n📸  Capture started for: {student_name} (ID: {student_id})")
print(f"📋  {len(ANGLES)} angles × {IMAGES_PER_ANGLE} photos = {total_images} total images")
print("     Press  C  to start each angle,  Q  to quit.\n")

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
angle_idx       = 0       # which angle we are on
count           = 0       # photos saved for current angle
capturing       = False   # actively saving frames?
waiting         = False   # in countdown phase?
wait_start      = 0.0
last_save_time  = 0.0

while angle_idx < len(ANGLES):
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Frame grab failed — retrying.")
        continue

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    now = time.time()

    # ── Draw face rectangles ──
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ── Determine status text ──
    if waiting:
        remaining = max(0, PREPARE_COUNTDOWN - int(now - wait_start))
        status_line1 = f"Get ready... {remaining}s"
        status_line2 = ANGLES[angle_idx]
        color = (0, 200, 255)
    elif capturing:
        status_line1 = f"Capturing {ANGLES[angle_idx]}  [{count}/{IMAGES_PER_ANGLE}]"
        status_line2 = f"Student: {student_name}  |  Angle {angle_idx + 1}/{len(ANGLES)}"
        color = (0, 255, 0)
    else:
        status_line1 = f"Angle {angle_idx + 1}/{len(ANGLES)}: {ANGLES[angle_idx]}"
        status_line2 = "Press  C  to start capturing"
        color = (0, 255, 255)

    draw_overlay(frame, status_line1, status_line2, color)

    # ── Countdown → capture transition ──
    if waiting and (now - wait_start >= PREPARE_COUNTDOWN):
        waiting    = False
        capturing  = True
        last_save_time = 0.0
        print(f"▶  Capturing angle {angle_idx + 1}: {ANGLES[angle_idx]}")

    # ── Save frames while capturing ──
    if capturing and len(faces) > 0:
        x, y, w, h = faces[0]   # use the first (largest) detected face
        face_roi = gray[y: y + h, x: x + w]

        if is_sharp(face_roi) and (now - last_save_time >= CAPTURE_DELAY_SEC):
            face_resized = cv2.resize(face_roi, FACE_SAVE_SIZE)
            filename = os.path.join(save_dir, f"{angle_idx}_{count + 1}.jpg")
            cv2.imwrite(filename, face_resized)
            count         += 1
            last_save_time = now

        # Check completion of this angle
        if count >= IMAGES_PER_ANGLE:
            print(f"✅  Angle {angle_idx + 1} done — {count} images saved.")
            angle_idx  += 1
            count       = 0
            capturing   = False

    cv2.imshow("AutoAttender — Face Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and not capturing and not waiting and angle_idx < len(ANGLES):
        waiting    = True
        wait_start = now
        print(f"⏳  Prepare for angle {angle_idx + 1}: {ANGLES[angle_idx]}")

    if key == ord('q'):
        print("⚠️  Capture aborted by user.")
        break

# ──────────────────────────────────────────────
# CLEANUP
# ──────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

saved = angle_idx * IMAGES_PER_ANGLE + count
print(f"\n✅  Capture complete — {saved} images saved to  {save_dir}/")
print("    Run  train.py  next to update the recognition model.")