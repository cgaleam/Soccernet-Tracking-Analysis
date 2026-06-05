"""
summarize_hyperparam.py — Lee todos los resultados de hiperparámetros
y genera una tabla comparativa en CSV y una gráfica de barras por parámetro.

Uso:
    python src/hyperparameter_study/summarize_hyperparam.py

Salida:
    evaluation_hyperparam/summary_table.csv
    evaluation_hyperparam/plots/
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Para Mac sin display

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
EVALUATION_BASE = os.path.join(BASE_DIR, '..', '..', 'evaluation_hyperparam', 'ocsort')
OUTPUT_CSV      = os.path.join(EVALUATION_BASE, 'summary_table.csv')
PLOTS_DIR       = os.path.join(EVALUATION_BASE, 'plots')

os.makedirs(PLOTS_DIR, exist_ok=True)

# Parámetros base de referencia
BASE_PARAMS = {
    'det_thresh'    : 0.3,
    'max_age'       : 30,
    'min_hits'      : 3,
    'iou_threshold' : 0.3,
}

# LEee resultados
def read_summary_csv(filepath):
    """Lee el CSV de summary y devuelve las métricas de la fila COMBINED."""
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('seq') == 'COMBINED' or row.get('') == 'COMBINED':
                return row
    return None

results = []

for folder in sorted(os.listdir(EVALUATION_BASE)):
    folder_path = os.path.join(EVALUATION_BASE, folder)
    if not os.path.isdir(folder_path) or folder == 'plots':
        continue

    # Buscar el CSV dentro de la subcarpeta del tracker
    csv_path = None
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f == 'pedestrian_detailed.csv':
                csv_path = os.path.join(root, f)
                break

    if not csv_path:
        print(f"  [!] No se encontró CSV en {folder}")
        continue

    row = read_summary_csv(csv_path)
    if not row:
        print(f"  [!] No se encontró fila COMBINED en {folder}")
        continue

    # Extraer nombre del parámetro y valor del nombre de la carpeta
    # Formato: ocsort_det_thresh_0.3
    parts = folder.replace('ocsort_', '').rsplit('_', 1)
    if len(parts) == 2:
        param_name = parts[0]
        param_value = parts[1]
    else:
        param_name = folder
        param_value = '?'

    # Extraer métricas clave
    try:
        hota = float(row.get('HOTA(0)', row.get('HOTA___50', 0))) * 100
        mota = float(row.get('MOTA', 0)) * 100
        idf1 = float(row.get('IDF1', 0)) * 100
        idsw = int(float(row.get('IDSW', 0)))
    except (ValueError, TypeError):
        print(f"  [!] Error leyendo métricas de {folder}")
        continue

    results.append({
        'tracker'    : folder,
        'param'      : param_name,
        'value'      : param_value,
        'HOTA'       : round(hota, 3),
        'MOTA'       : round(mota, 3),
        'IDF1'       : round(idf1, 3),
        'IDSW'       : idsw
    })

    print(f"  {folder}: HOTA={hota:.2f} MOTA={mota:.2f} IDF1={idf1:.2f}")

# GUARDAR CSV RESUMEN
if results:
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tracker', 'param', 'value', 'HOTA', 'MOTA', 'IDF1', 'IDSW'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nTabla guardada en: {OUTPUT_CSV}")
else:
    print("No se encontraron resultados.")
    exit()


# GENERAR GRÁFICAS POR PARÁMETRO
params = list(set(r['param'] for r in results))

for param in params:
    param_results = [r for r in results if r['param'] == param]
    param_results.sort(key=lambda x: float(x['value']) if x['value'] != '?' else 999)

    values = [str(r['value']) for r in param_results]
    hotas  = [r['HOTA'] for r in param_results]
    motas  = [r['MOTA'] for r in param_results]
    idf1s  = [r['IDF1'] for r in param_results]

    x = range(len(values))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width for i in x], hotas, width, label='HOTA', color='#2196F3')
    bars2 = ax.bar([i for i in x],         motas, width, label='MOTA', color='#4CAF50')
    bars3 = ax.bar([i + width for i in x], idf1s, width, label='IDF1', color='#FF9800')

    # Marcar el valor base
    base_value = str(BASE_PARAMS.get(param, ''))
    if base_value in values:
        base_idx = values.index(base_value)
        ax.axvline(x=base_idx, color='red', linestyle='--', alpha=0.5, label=f'Base ({base_value})')

    ax.set_xlabel(f'Valor de {param}')
    ax.set_ylabel('Score (%)')
    ax.set_title(f'OCSORT — Efecto de {param} sobre métricas (test)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(values)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Añadir valores encima de las barras
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f'ocsort_{param}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Gráfica guardada: {plot_path}")

print("\nResumen completado.")
print(f"CSV:     {OUTPUT_CSV}")
print(f"Gráficas: {PLOTS_DIR}")