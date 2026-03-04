import cv2
import os
import numpy as np

dataset_path = "dataset"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}

current_label = 0

people = sorted(os.listdir(dataset_path))
print("Found people folders:", people)

for person in people:
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    print(f"\nProcessing person: {person}")
    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("  Skipped unreadable image:", img_path)
            continue

        faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

        if len(faces_detected) == 0:
            # Images captured by capture.py are already cropped face-only,
            # so use the full image directly if no face is detected
            face_roi = cv2.resize(img, (100, 100))  # FIXED: was (200, 200)
            faces.append(face_roi)
            labels.append(current_label)
            print("  Added (full crop):", img_name)
        else:
            for (x, y, w, h) in faces_detected:
                face_roi = cv2.resize(img[y:y+h, x:x+w], (100, 100))  # FIXED: was (200, 200)
                faces.append(face_roi)
                labels.append(current_label)
                print("  Added face from:", img_name)

    current_label += 1

if len(faces) == 0:
    print("\nNo faces found in dataset. Training aborted.")
else:
    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")
    np.save("labels.npy", label_map)  # FIXED: was missing — labels.npy never saved
    print("\n✅ Training completed successfully.")
    print(f"   Total faces trained: {len(faces)}")
    print(f"   Label Map: {label_map}")