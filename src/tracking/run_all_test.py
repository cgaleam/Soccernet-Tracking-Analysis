"""
run_all_test.py — Ejecuta todos los algoritmos sobre todas las secuencias de test
y guarda los resultados en la carpeta results_test/

Uso:
    python src/run_all_test.py
"""

import os
import numpy as np
import cv2
import random
from boxmot import BYTETracker, OCSORT

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
DATASET_PATH = "../SoccerNet/tracking/test"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', '..',  'results_test')

# Parámetros de detección del balón
BALL_MAX_AREA  = 2000
BALL_MIN_RATIO = 0.7
BALL_MAX_RATIO = 1.3

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

def load_detections(det_path):
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
    return detections

def is_ball(x, y, w, h):
    area  = w * h
    ratio = w / h if h > 0 else 1.0
    return area < BALL_MAX_AREA and BALL_MIN_RATIO < ratio < BALL_MAX_RATIO

def run_tracker(tracker_factory, tracker_name, sequence_path, results_dir, show=False):
    tracker     = tracker_factory()
    sequence    = os.path.basename(sequence_path.rstrip('/\\'))
    img_folder  = os.path.join(sequence_path, "img1")
    det_path    = os.path.join(sequence_path, "det", "det.txt")
    result_path = os.path.join(results_dir, f"{sequence}.txt")

    if not os.path.exists(det_path):
        print(f"  [!] Sin det.txt en {sequence}, saltando...")
        return

    detections = load_detections(det_path)
    images     = sorted(os.listdir(img_folder))
    results    = []

    for i, img_name in enumerate(images, start=1):
        frame = cv2.imread(os.path.join(img_folder, img_name))
        if frame is None:
            continue

        dets_frame  = detections.get(i, [])
        player_dets = []

        for (x, y, w, h, conf) in dets_frame:
            if not is_ball(x, y, w, h):
                player_dets.append([x, y, x + w, y + h, conf, 0])

        dets_np = np.array(player_dets, dtype=float) if player_dets else np.empty((0, 6))

        try:
            tracks = tracker.update(dets_np, frame)
        except TypeError:
            tracks = np.empty((0, 5))

        for track in tracks:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            w_out = x2 - x1
            h_out = y2 - y1
            results.append(f"{i},{int(track_id)},{x1},{y1},{w_out},{h_out},1,-1,-1,-1")

    os.makedirs(results_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        f.write('\n'.join(results))

    print(f"  [OK] {sequence} → {result_path} ({len(results)} tracks)")

# Main
if __name__ == "__main__":

    sequences = sorted([
        os.path.join(DATASET_PATH, d)
        for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])

    print(f"Secuencias test encontradas: {len(sequences)}")

    trackers = [
        {
            "name": "bytetracker",
            "factory": lambda: BYTETracker(),
            "results_dir": os.path.join(RESULTS_PATH, "bytetracker")
        },
        {
            "name": "ocsort",
            "factory": lambda: OCSORT(
                det_thresh=0.3,
                max_age=30,
                min_hits=3,
                iou_threshold=0.3,
                delta_t=3,
                asso_func='iou',
                inertia=0.2
            ),
            "results_dir": os.path.join(RESULTS_PATH, "ocsort")
        }
    ]

    for tracker_cfg in trackers:
        print(f"\n{'='*50}")
        print(f"Ejecutando: {tracker_cfg['name'].upper()} en TEST")
        print(f"{'='*50}")

        for seq_path in sequences:
            seq_name = os.path.basename(seq_path)
            print(f"  Procesando {seq_name}...")
            run_tracker(
                tracker_factory=tracker_cfg["factory"],
                tracker_name=tracker_cfg["name"],
                sequence_path=seq_path,
                results_dir=tracker_cfg["results_dir"],
                show=False
            )

        print(f"\n{tracker_cfg['name'].upper()} TEST completado.")

    print(f"\n{'='*50}")
    print("Todos los algoritmos completados en TEST.")
    print(f"Resultados en: {RESULTS_PATH}")
    print(f"{'='*50}")