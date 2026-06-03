"""
run_all.py — Ejecuta todos los algoritmos sobre todas las secuencias de SoccerNet
y guarda los resultados en la carpeta results/

Uso:
    python src/run_all.py

Estructura de resultados generada:
    results/
    ├── bytetracker/
    │   ├── SNMOT-060.txt
    │   ├── SNMOT-061.txt
    │   └── ...
    └── ocsort/
        ├── SNMOT-060.txt
        ├── SNMOT-061.txt
        └── ...
"""

import os
import sys
import numpy as np
import cv2
import random
from boxmot import BYTETracker, OCSORT

# ─────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────
DATASET_PATH = "../SoccerNet/tracking/train/"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'results_train')

# ─────────────────────────────────────────
# PARÁMETROS DE DETECCIÓN DEL BALÓN
# ─────────────────────────────────────────
BALL_MAX_AREA  = 2000
BALL_MIN_RATIO = 0.7
BALL_MAX_RATIO = 1.3

# ─────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────
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
    """Ejecuta un tracker sobre una secuencia y guarda los resultados."""
    tracker     = tracker_factory()   # instancia nueva por secuencia
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
        tracks  = tracker.update(dets_np, frame)

        for track in tracks:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            w_out = x2 - x1
            h_out = y2 - y1
            # Formato MOT: frame, id, x, y, w, h, conf, -1, -1, -1
            results.append(f"{i},{int(track_id)},{x1},{y1},{w_out},{h_out},1,-1,-1,-1")

        if show:
            for track in tracks:
                x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
                color = get_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"{tracker_name} | {sequence} | Frame {i}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break

    # Guardar resultados
    os.makedirs(results_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        f.write('\n'.join(results))

    print(f"  [OK] {sequence} → {result_path} ({len(results)} tracks)")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Obtener todas las secuencias disponibles
    sequences = sorted([
        os.path.join(DATASET_PATH, d)
        for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])

    print(f"Secuencias encontradas: {len(sequences)}")

    # Definir algoritmos a ejecutar
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

    # Ejecutar cada algoritmo sobre todas las secuencias
    for tracker_cfg in trackers:
        print(f"\n{'='*50}")
        print(f"Ejecutando: {tracker_cfg['name'].upper()}")
        print(f"{'='*50}")

        for seq_path in sequences:
            seq_name = os.path.basename(seq_path)
            print(f"  Procesando {seq_name}...")
            run_tracker(
                tracker_factory=tracker_cfg["factory"],
                tracker_name=tracker_cfg["name"],
                sequence_path=seq_path,
                results_dir=tracker_cfg["results_dir"],
                show=False  # Cambia a True si quieres ver el vídeo mientras procesa
            )

        print(f"\n{tracker_cfg['name'].upper()} completado.")

    print(f"\n{'='*50}")
    print("Todos los algoritmos completados.")
    print(f"Resultados en: {RESULTS_PATH}")
    print(f"{'='*50}")