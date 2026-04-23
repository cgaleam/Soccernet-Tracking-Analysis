import cv2
import os

# paths
base_path = r"path/to/SoccerNet/tracking/train/SNMOT-105/"
img_folder = os.path.join(base_path, "img1")
gt_path = os.path.join(base_path, "gt/gt.txt")

# Detectar balon con YOLOv8
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Para asignar colores únicos a cada ID o jugador
import random

colors = {}

def get_color(id):
    if id not in colors:
        colors[id] = (
            random.randint(0,255),
            random.randint(0,255),
            random.randint(0,255)
        )
    return colors[id]

# Leer detecciones
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

# Mostrar frames
images = sorted(os.listdir(img_folder))

for i, img_name in enumerate(images, start=1):
    frame = cv2.imread(os.path.join(img_folder, img_name))

    if i in gt:
        for (obj_id, x, y, w, h) in gt[i]:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            # 🔥 Dibujar ID
            cv2.putText(frame, f"ID {obj_id}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Tracking con ID", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()