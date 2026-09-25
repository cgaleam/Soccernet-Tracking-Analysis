"""
hyperparam_ocsort.py — Estudio de hiperparámetros para OCSORT
Varía un parámetro a la vez manteniendo los demás fijos y evalúa con trackeval.

Resultados en:
    evaluation_hyperparam/ocsort/
"""

import os
import numpy as np
import cv2
import trackeval
from boxmot import OCSORT

# ─────────────────────────────────────────
# CONFIGURACIÓNç
# ─────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_TRAIN   = os.path.join(BASE_DIR, '..', '..', '..', 'SoccerNet', 'tracking', 'train')
DATASET_TEST    = os.path.join(BASE_DIR, '..', '..', '..', 'SoccerNet', 'tracking', 'test')
RESULTS_BASE    = os.path.join(BASE_DIR, '..', '..', 'results_hyperparam', 'ocsort')
EVALUATION_BASE = os.path.join(BASE_DIR, '..', '..', 'evaluation_hyperparam', 'ocsort')
# Hiperparámetros base
BASE_PARAMS = {
    'det_thresh'    : 0.3,
    'max_age'       : 30,
    'min_hits'      : 3,
    'iou_threshold' : 0.3,
    'delta_t'       : 3,
    'asso_func'     : 'iou',
    'inertia'       : 0.2
}

# Valores a probar para cada hiperparámetro
SEARCH_SPACE = {
    'det_thresh'    : [0.1, 0.2, 0.3, 0.4, 0.5],
    'max_age'       : [10, 20, 30, 50],
    'min_hits'      : [1, 2, 3, 5],
    'iou_threshold' : [0.1, 0.2, 0.3, 0.4, 0.5],
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

def run_tracker(params, sequence_path, results_dir):
    tracker  = OCSORT(**params)
    sequence = os.path.basename(sequence_path.rstrip('/\\'))
    img_folder  = os.path.join(sequence_path, "img1")
    det_path    = os.path.join(sequence_path, "det", "det.txt")
    result_path = os.path.join(results_dir, f"{sequence}.txt")

    if not os.path.exists(det_path):
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

def evaluate(results_path, dataset_path, split, output_path, tracker_name):
    os.makedirs(output_path, exist_ok=True)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    eval_config['OUTPUT_FOLDER']         = output_path
    eval_config['PRINT_RESULTS']         = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER']          = dataset_path
    dataset_config['TRACKERS_FOLDER']    = results_path
    dataset_config['SPLIT_TO_EVAL']      = split
    dataset_config['BENCHMARK']          = 'SoccerNet'
    dataset_config['OUTPUT_FOLDER']      = output_path
    dataset_config['TRACKERS_TO_EVAL']   = [tracker_name]
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    dataset_config['SKIP_SPLIT_FOL']     = True
    dataset_config['DO_PREPROC']         = False

    metrics_config = {'METRICS': ['HOTA', 'MOTA', 'IDF1'], 'THRESHOLD': 0.5}

    evaluator    = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(metrics_config),
                    trackeval.metrics.CLEAR(metrics_config),
                    trackeval.metrics.Identity(metrics_config)]

    results = evaluator.evaluate(dataset_list, metrics_list)
    return results

# Main
if __name__ == "__main__":

    sequences = sorted([
        os.path.join(DATASET_TEST, d)
        for d in os.listdir(DATASET_TEST)
        if os.path.isdir(os.path.join(DATASET_TEST, d))
    ])

    print(f"Secuencias test: {len(sequences)}")
    print("="*60)

    summary = []

    for param_name, values in SEARCH_SPACE.items():
        print(f"\nVariando: {param_name}")
        print("-"*40)

        for value in values:
            # Crear configuración variando solo este parámetro
            params = BASE_PARAMS.copy()
            params[param_name] = value

            tracker_name = f"ocsort_{param_name}_{value}"
            results_dir  = os.path.join(RESULTS_BASE, tracker_name)
            eval_dir     = os.path.join(EVALUATION_BASE, tracker_name)

            print(f"  {param_name}={value} → ejecutando...")

            # Ejecutar sobre todas las secuencias de test
            for seq_path in sequences:
                run_tracker(params, seq_path, results_dir)

            # Evaluar
            evaluate(RESULTS_BASE, DATASET_TEST, 'test', eval_dir, tracker_name)

            # Leer HOTA del summary CSV
            summary_csv = os.path.join(eval_dir, tracker_name,
                                       'pedestrian_summary.txt')
            print(f"  {param_name}={value} → completado. Resultados en {eval_dir}")

            summary.append({
                'param': param_name,
                'value': value,
                'results_dir': eval_dir
            })

    print("\n" + "="*60)
    print("Estudio de hiperparámetros completado.")
    print(f"Todos los resultados en: {EVALUATION_BASE}/")
    print("="*60)