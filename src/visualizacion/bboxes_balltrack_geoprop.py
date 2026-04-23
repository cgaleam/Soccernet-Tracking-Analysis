import cv2
import os
import random

# paths
base_path = r"path/to/SoccerNet/tracking/train/SNMOT-105/"
img_folder = os.path.join(base_path, "img1")
gt_path = os.path.join(base_path, "gt/gt.txt")

colors = {}

def get_color(track_id):
    if track_id not in colors:
        colors[track_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
    return colors[track_id]

def is_ball(w, h, median_area):
    area = w * h
    ratio = w / h  # el balón es casi cuadrado (ratio ≈ 1)
    # El balón ocupa menos del 15% del área mediana del frame
    # y su bbox es aproximadamente cuadrada
    return area < 0.15 * median_area and 0.5 < ratio < 2.0

# Leer ground truth
gt = {}
with open(gt_path) as f:
    for line in f:
        parts = line.strip().split(',')
        frame_id = int(parts[0])
        obj_id = int(parts[1])
        x, y, w, h = map(int, parts[2:6])
        if frame_id not in gt:
            gt[frame_id] = []
        gt[frame_id].append((obj_id, x, y, w, h))

images = sorted(os.listdir(img_folder))

for i, img_name in enumerate(images, start=1):
    frame = cv2.imread(os.path.join(img_folder, img_name))
    if frame is None:
        continue

    if i in gt:
        detections = gt[i]

        # Umbral adaptativo: mediana de áreas en este frame
        areas = sorted(w * h for (_, _, _, w, h) in detections)
        median_area = areas[len(areas) // 2] if areas else 1

        for (obj_id, x, y, w, h) in detections:
            if is_ball(w, h, median_area):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "BALL", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                color = get_color(obj_id)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID {obj_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Tracking con balón", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()
