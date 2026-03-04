"""
Module 2 — Train LBPH Face Recognizer
=======================================
Reads all face images from  dataset/  and trains an LBPH model.

Output files (written to project root so recognize.py can find them):
    trainer.yml   — LBPH model weights
    labels.npy    — dict mapping integer label → student name

Usage:
    python train.py
"""

import cv2
import os
import numpy as np

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATASET_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
MODEL_OUT    = os.path.join(os.path.dirname(__file__), "..", "trainer.yml")
LABELS_OUT   = os.path.join(os.path.dirname(__file__), "..", "labels.npy")
FACE_SIZE    = (100, 100)   # must match capture_face.py → FACE_SAVE_SIZE

# ──────────────────────────────────────────────
# LOAD HAAR CASCADE  (used only if images aren't pre-cropped)
# ──────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ──────────────────────────────────────────────
# COLLECT TRAINING DATA
# ──────────────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces:     list = []
labels:    list = []
label_map: dict = {}          # {int_label: student_name}
next_label: int = 0

dataset_path = os.path.abspath(DATASET_DIR)
if not os.path.isdir(dataset_path):
    print(f"❌  Dataset folder not found: {dataset_path}")
    exit(1)

people = sorted(
    p for p in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, p))
)

if not people:
    print("❌  No student folders found inside dataset/. Run capture_face.py first.")
    exit(1)

print(f"📂  Found {len(people)} student folder(s): {people}\n")

for person_folder in people:
    person_path = os.path.join(dataset_path, person_folder)

    # Try to read the human-readable name saved by capture_face.py
    name_file = os.path.join(person_path, "name.txt")
    if os.path.exists(name_file):
        with open(name_file) as nf:
            display_name = nf.read().strip() or person_folder
    else:
        display_name = person_folder   # fall back to folder name

    print(f"  Processing: {display_name}  (folder: {person_folder})")
    label_map[next_label] = display_name
    person_faces = 0

    for img_name in sorted(os.listdir(person_path)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"    ⚠️  Skipped unreadable file: {img_name}")
            continue

        # Images from capture_face.py are already 100×100 cropped faces.
        # Run detection only if the image is not pre-cropped (unexpected size).
        if img.shape == FACE_SIZE:
            face_roi = img
        else:
            detected = face_cascade.detectMultiScale(img, 1.1, 5)
            if len(detected) == 0:
                face_roi = cv2.resize(img, FACE_SIZE)
            else:
                x, y, w, h = detected[0]
                face_roi = cv2.resize(img[y: y + h, x: x + w], FACE_SIZE)

        # ⚠️ CRITICAL: apply the same equalizeHist used in recognize.py so
        #    the model is trained on identical-looking pixel data.
        face_roi = cv2.equalizeHist(face_roi)
        faces.append(face_roi)
        labels.append(next_label)
        person_faces += 1

    print(f"    → {person_faces} face samples loaded.")
    next_label += 1

# ──────────────────────────────────────────────
# TRAIN AND SAVE
# ──────────────────────────────────────────────
print()
if len(faces) == 0:
    print("❌  No face samples found. Training aborted.")
    exit(1)

print(f"🧠  Training LBPH model on {len(faces)} face samples across {len(label_map)} student(s) …")
recognizer.train(faces, np.array(labels))

model_path  = os.path.abspath(MODEL_OUT)
labels_path = os.path.abspath(LABELS_OUT)

recognizer.save(model_path)
np.save(labels_path, label_map)

print(f"\n✅  Training complete!")
print(f"     Model   → {model_path}")
print(f"     Labels  → {labels_path}")
print(f"     Samples → {len(faces)}")
print(f"     Students: {list(label_map.values())}")
print("\n    Run  recognize.py  next to start attendance.")
