"""
ocsort_optimal.py — Ejecuta y evalúa OCSORT con la configuración óptima
encontrada en el estudio de hiperparámetros.

Configuración óptima:
    iou_threshold = 0.1  (mayor impacto, HOTA 80.17 vs 75.51 base)
    max_age       = 10   (mejor que 30 base)
    min_hits      = 1    (mejor HOTA aunque más IDSW)

Uso:
    python src/hyperparameter_study/ocsort_optimal.py

Resultados en:
    results_hyperparam/ocsort_optimal/
    evaluation_hyperparam/ocsort_optimal/
"""

import os
import numpy as np
import cv2
import trackeval
from boxmot import OCSORT

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_TEST    = os.path.join(BASE_DIR, '..', '..', '..', 'SoccerNet', 'tracking', 'test')
RESULTS_DIR    = os.path.join(BASE_DIR, '..', '..', 'results_hyperparam', 'ocsort', 'ocsort_optimal')
EVALUATION_DIR = os.path.join(BASE_DIR, '..', '..', 'evaluation_hyperparam', 'ocsort', 'ocsort_optimal')
TRACKER_NAME    = 'ocsort_optimal'

# ─────────────────────────────────────────
# PARÁMETROS ÓPTIMOS
# ─────────────────────────────────────────
OPTIMAL_PARAMS = {
    'det_thresh'    : 0.3,   # no afecta al rendimiento
    'max_age'       : 10,    # óptimo encontrado
    'min_hits'      : 1,     # óptimo encontrado
    'iou_threshold' : 0.1,   # mayor impacto positivo
    'delta_t'       : 3,
    'asso_func'     : 'iou',
    'inertia'       : 0.2
}

# ─────────────────────────────────────────
# PARÁMETROS DE DETECCIÓN DEL BALÓN
# ─────────────────────────────────────────
BALL_MAX_AREA  = 2000
BALL_MIN_RATIO = 0.7
BALL_MAX_RATIO = 1.3


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

def run_tracker(sequence_path):
    tracker  = OCSORT(**OPTIMAL_PARAMS)
    sequence = os.path.basename(sequence_path.rstrip('/\\'))
    img_folder  = os.path.join(sequence_path, "img1")
    det_path    = os.path.join(sequence_path, "det", "det.txt")
    result_path = os.path.join(RESULTS_DIR, f"{sequence}.txt")

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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(result_path, 'w') as f:
        f.write('\n'.join(results))

    print(f"  [OK] {sequence} ({len(results)} tracks)")

# ─────────────────────────────────────────
# EVALUACIÓN CON TRACKEVAL
# ─────────────────────────────────────────
def evaluate():
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    eval_config['OUTPUT_FOLDER']         = EVALUATION_DIR
    eval_config['PRINT_RESULTS']         = True

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER']          = DATASET_TEST
    dataset_config['TRACKERS_FOLDER']    = os.path.join(BASE_DIR, '..', '..', 'results_hyperparam', 'ocsort')
    dataset_config['SPLIT_TO_EVAL']      = 'test'
    dataset_config['BENCHMARK']          = 'SoccerNet'
    dataset_config['OUTPUT_FOLDER']      = EVALUATION_DIR
    dataset_config['TRACKERS_TO_EVAL']   = [TRACKER_NAME]
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    dataset_config['SKIP_SPLIT_FOL']     = True
    dataset_config['DO_PREPROC']         = False

    metrics_config = {'METRICS': ['HOTA', 'MOTA', 'IDF1'], 'THRESHOLD': 0.5}

    evaluator    = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config)
    ]

    evaluator.evaluate(dataset_list, metrics_list)

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    sequences = sorted([
        os.path.join(DATASET_TEST, d)
        for d in os.listdir(DATASET_TEST)
        if os.path.isdir(os.path.join(DATASET_TEST, d))
    ])

    print("="*60)
    print("OCSORT — Configuración óptima")
    print(f"Parámetros: {OPTIMAL_PARAMS}")
    print(f"Secuencias test: {len(sequences)}")
    print("="*60)

    for seq_path in sequences:
        seq_name = os.path.basename(seq_path)
        print(f"  Procesando {seq_name}...")
        run_tracker(seq_path)

    print("\nTracking completado. Evaluando...")
    evaluate()

    print("\n" + "="*60)
    print("OCSORT óptimo completado.")
    print(f"Resultados en: {EVALUATION_DIR}")
    print("="*60)