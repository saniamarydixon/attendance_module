# AutoAttender

Automatic face-recognition attendance system using OpenCV LBPH.

## Three-Step Pipeline

### Step 1 — Register a student
```
cd d:\S6_PROJECT\proj\autoattender
python capture_face.py
```
- Enter the student's **ID** and **Name** when prompted.
- Follow the on-screen angle instructions (**Straight → Left → Right**).
- Press **C** to start capturing each angle.
- 30 images are saved per angle (90 total) under `dataset/<student_id>/`.
- Repeat for every student you want to register.

### Step 2 — Train the model
```
python train.py
```
- Reads all images from `dataset/` and trains a new LBPH model.
- Outputs `trainer.yml` and `labels.npy` in the project root.
- **Run this once after registering all students** (or again whenever you add a new student).

### Step 3 — Run attendance
```
python recognize.py
```
- Opens the webcam and recognises registered faces in real time.
- Tracks how long each student remains on screen.
- Press **Q** to quit — results are saved to `attendance.csv`.

## File Layout
```
autoattender/
  capture_face.py   ← Step 1: capture & register
  train.py          ← Step 2: train model
  recognize.py      ← Step 3: live recognition + attendance

dataset/
  <student_id>/     ← grayscale face images + name.txt

trainer.yml         ← trained LBPH model (auto-generated)
labels.npy          ← label→name mapping  (auto-generated)
attendance.csv      ← attendance records  (appended each session)
```

## Attendance Rules
| Rule | Value |
|------|-------|
| Minimum on-screen time to be marked **Present** | 60 seconds |
| LBPH confidence threshold | ≤ 100 (lower = better match) |
| Frames needed to confirm identity | 5 consecutive frames |

## Requirements
```
pip install opencv-python opencv-contrib-python numpy
```
