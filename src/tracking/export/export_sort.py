"""
export_ocsort.py — Exporta un vídeo de tracking con OCSORT
El vídeo se guarda en la carpeta videos/ en la raíz del proyecto.

Uso:
    python src/tracking/export/export_ocsort.py
"""

import cv2
import os
import numpy as np
import random
from boxmot import OCSORT

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
DATASET_PATH = r"../path/to/SoccerNet/tracking/train"
SEQUENCE     = "SNMOT-076"
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'videos')

base_path   = os.path.join(DATASET_PATH, SEQUENCE)
img_folder  = os.path.join(base_path, "img1")
det_path    = os.path.join(base_path, "det", "det.txt")
output_path = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_ocsort.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# PARÁMETROS DE DETECCIÓN DEL BALÓN
# ─────────────────────────────────────────
BALL_MAX_AREA  = 2000
BALL_MIN_RATIO = 0.7
BALL_MAX_RATIO = 1.3

# REMAPEO DE IDs → 1..N
id_map  = {}
next_id = [1]

def get_sequential_id(tracker_id):
    if tracker_id not in id_map:
        id_map[tracker_id] = next_id[0]
        next_id[0] += 1
    return id_map[tracker_id]


colors = {}

def get_color(track_id):
    if track_id not in colors:
        random.seed(int(track_id) * 7)
        colors[track_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return colors[track_id]

# CARGAR DETECCIONES (det.txt)
detections = {}

with open(det_path) as f:
    for line in f:
        parts = line.strip().split(',')
        frame_id = int(parts[0])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        conf = float(parts[6]) if len(parts) > 6 else 1.0
        if frame_id not in detections:
            detections[frame_id] = []
        detections[frame_id].append((x, y, w, h, conf))

# ─────────────────────────────────────────
# INICIALIZAR TRACKER Y VIDEO WRITER
# ─────────────────────────────────────────
tracker = OCSORT(
    det_thresh=0.3,
    max_age=30,
    min_hits=3,
    iou_threshold=0.3,
    delta_t=3,
    asso_func='iou',
    inertia=0.2
)

images = sorted(os.listdir(img_folder))

# Leer primer frame para obtener dimensiones
first_frame = cv2.imread(os.path.join(img_folder, images[0]))
h_vid, w_vid = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, 25.0, (w_vid, h_vid))

print(f"Exportando {SEQUENCE} con OCSORT...")
print(f"Resolución: {w_vid}x{h_vid} | Frames: {len(images)}")

# ─────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────
for i, img_name in enumerate(images, start=1):
    frame = cv2.imread(os.path.join(img_folder, img_name))
    if frame is None:
        continue

    dets_frame  = detections.get(i, [])
    ball_dets   = []
    player_dets = []

    for (x, y, w, h, conf) in dets_frame:
        area  = w * h
        ratio = w / h if h > 0 else 1.0
        if area < BALL_MAX_AREA and BALL_MIN_RATIO < ratio < BALL_MAX_RATIO:
            ball_dets.append((x, y, w, h))
        else:
            player_dets.append([x, y, x + w, y + h, conf, 0])

    dets_np = np.array(player_dets, dtype=float) if player_dets else np.empty((0, 6))
    try:
        tracks = tracker.update(dets_np, frame)
    except TypeError:
        tracks = np.empty((0, 5))


    # ── Dibujar jugadores ──
    for track in tracks:
        x1, y1, x2, y2, raw_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
        track_id = get_sequential_id(raw_id)
        color = get_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ── Dibujar balón ──
    for (x, y, w, h) in ball_dets:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "BALL", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ── Info en pantalla ──
    cv2.putText(frame, f"OCSORT | {SEQUENCE} | Frame {i}/{len(images)}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Tracks activos: {len(tracks)}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    writer.write(frame)

    if i % 100 == 0:
        print(f"  Frame {i}/{len(images)}...")

writer.release()
print(f"\nVídeo exportado en: {output_path}")