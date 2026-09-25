"""
evaluate_test.py — Evalúa los resultados de tracking sobre el split test
Calcula HOTA, MOTA e IDF1 para cada algoritmo.

Uso:
    python src/evaluation/evaluate_test.py
"""

import sys
import os

DATASET_PATH    = "../SoccerNet/tracking/test"


import trackeval

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
RESULTS_PATH    = os.path.join(os.path.dirname(__file__), '..', '..', 'results_test')
EVALUATION_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'evaluation_test')

os.makedirs(EVALUATION_PATH, exist_ok=True)

# Configuración del evaluador
eval_config = trackeval.Evaluator.get_default_eval_config()
eval_config['DISPLAY_LESS_PROGRESS'] = False
eval_config['OUTPUT_FOLDER']         = EVALUATION_PATH
eval_config['PRINT_RESULTS']         = True
eval_config['PRINT_ONLY_COMBINED']   = False

# Configuración del dataset
dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dataset_config['GT_FOLDER']          = DATASET_PATH
dataset_config['TRACKERS_FOLDER']    = RESULTS_PATH
dataset_config['SPLIT_TO_EVAL']      = 'test'
dataset_config['BENCHMARK']          = 'SoccerNet'
dataset_config['OUTPUT_FOLDER']      = EVALUATION_PATH
dataset_config['TRACKERS_TO_EVAL']   = ['bytetracker', 'ocsort']
dataset_config['TRACKER_SUB_FOLDER'] = ''
dataset_config['SKIP_SPLIT_FOL']     = True
dataset_config['DO_PREPROC']         = False

# Métricas a calcular
metrics_config = {
    'METRICS': ['HOTA', 'MOTA', 'IDF1'],
    'THRESHOLD': 0.5
}

# Ejecutar evaluación
print("="*60)
print("Iniciando evaluación TEST con TrackEval")
print(f"Dataset: {DATASET_PATH}")
print(f"Resultados: {RESULTS_PATH}")
print(f"Salida: {EVALUATION_PATH}")
print("="*60)

evaluator    = trackeval.Evaluator(eval_config)
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
metrics_list = []

for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
    metrics_list.append(metric(metrics_config))

evaluator.evaluate(dataset_list, metrics_list)

print("\nEvaluación TEST completada.")
print(f"Resultados guardados en: {EVALUATION_PATH}")