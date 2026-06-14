"""
heatmap.py — Genera un mapa de calor de posiciones de jugadores
a partir de los resultados de tracking (configuración óptima de BYTETracker).

Las posiciones se aproximan usando el centro de cada bounding box en
coordenadas de píxel de la cámara. No se aplica homografía, por lo que
la proyección sobre el campo es aproximada debido a la perspectiva.

Uso:
    python src/metadata/heatmap.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Evita que se abra una ventana de visualización al generar el mapa de calor

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', '..', 'results_hyperparam', 'bytetracker', 'bytetracker_optimal')
OUTPUT_DIR  = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'heatmaps')

SEQUENCE = "SNMOT-118"  # secuencia a analizar

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CARGAR RESULTADOS DE TRACKING
result_path = os.path.join(RESULTS_DIR, f"{SEQUENCE}.txt")

if not os.path.exists(result_path):
    raise FileNotFoundError(f"No se encontró {result_path}")

x_centers = []
y_centers = []

with open(result_path) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

        # Centro del bounding box
        cx = x + w / 2
        cy = y + h / 2

        x_centers.append(cx)
        y_centers.append(cy)

print(f"Secuencia: {SEQUENCE}")
print(f"Total de posiciones: {len(x_centers)}")
print(f"Rango X: {min(x_centers):.0f} - {max(x_centers):.0f}")
print(f"Rango Y: {min(y_centers):.0f} - {max(y_centers):.0f}")

# GENERAR MAPA DE CALOR
fig, ax = plt.subplots(figsize=(12, 7))

# Heatmap 2D con hexbin
hb = ax.hexbin(x_centers, y_centers, gridsize=40, cmap='YlOrRd', mincnt=1)

# La imagen tiene origen (0,0) en la esquina superior izquierda
ax.invert_yaxis()

ax.set_title(f"Mapa de calor de posiciones — {SEQUENCE}\n"
              f"(BYTETracker, configuración óptima — posiciones en píxeles, sin homografía)")
ax.set_xlabel("Posición X (píxeles)")
ax.set_ylabel("Posición Y (píxeles)")

cbar = fig.colorbar(hb, ax=ax)
cbar.set_label("Densidad de posiciones")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_heatmap.png")
plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nMapa de calor guardado en: {output_path}")