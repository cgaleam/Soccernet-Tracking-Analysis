"""
evaluate.py — Evalúa los resultados de tracking con trackeval
Calcula HOTA, MOTA e IDF1 para cada algoritmo sobre todas las secuencias

Uso:
    python src/evaluation/evaluate.py
"""

import sys
import os

DATASET_PATH = r"C:\Ingeniería del Software\4\2do Cuatri\TFG\path\to\SoccerNet\tracking\train"

import trackeval

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
RESULTS_PATH    = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
EVALUATION_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'evaluation')

os.makedirs(EVALUATION_PATH, exist_ok=True)

# ─────────────────────────────────────────
# CONFIGURACIÓN DEL EVALUADOR
# ─────────────────────────────────────────
eval_config = trackeval.Evaluator.get_default_eval_config()
eval_config['DISPLAY_LESS_PROGRESS'] = False
eval_config['OUTPUT_FOLDER']         = EVALUATION_PATH
eval_config['PRINT_RESULTS']         = True
eval_config['PRINT_ONLY_COMBINED']   = False

# ─────────────────────────────────────────
# CONFIGURACIÓN DEL DATASET
# ─────────────────────────────────────────
dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dataset_config['GT_FOLDER']         = DATASET_PATH
dataset_config['TRACKERS_FOLDER']   = RESULTS_PATH
dataset_config['SPLIT_TO_EVAL']     = 'train'
dataset_config['BENCHMARK']         = 'SoccerNet'
dataset_config['OUTPUT_FOLDER']     = EVALUATION_PATH
dataset_config['TRACKERS_TO_EVAL']  = ['bytetracker', 'ocsort']
dataset_config['TRACKER_SUB_FOLDER'] = ''
dataset_config['SKIP_SPLIT_FOL'] = True
dataset_config['DO_PREPROC'] = False


# ─────────────────────────────────────────
# MÉTRICAS A CALCULAR
# ─────────────────────────────────────────
metrics_config = {
    'METRICS': ['HOTA', 'MOTA', 'IDF1'],
    'THRESHOLD': 0.5
}

# ─────────────────────────────────────────
# EJECUTAR EVALUACIÓN
# ─────────────────────────────────────────
print("="*60)
print("Iniciando evaluación con TrackEval")
print(f"Dataset: {DATASET_PATH}")
print(f"Resultados: {RESULTS_PATH}")
print(f"Salida: {EVALUATION_PATH}")
print("="*60)

evaluator = trackeval.Evaluator(eval_config)
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
metrics_list = []

for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
    metrics_list.append(metric(metrics_config))

evaluator.evaluate(dataset_list, metrics_list)

print("\nEvaluación completada.")
print(f"Resultados guardados en: {EVALUATION_PATH}")