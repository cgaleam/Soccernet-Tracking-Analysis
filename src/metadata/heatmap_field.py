"""
heatmap_field.py — Genera un mapa de calor de posiciones de jugadores
proyectado sobre un campo de fútbol dibujado en 2D (vista cenital).

Las posiciones del tracking (en píxeles de la cámara) se escalan usando
el rango real observado de posiciones y se proyectan sobre un
campo estándar de 105x68 metros. Se usa una
estimación de densidad (KDE) para obtener un degradado continuo que
cubre todo el campo, incluyendo zonas de baja densidad.

Uso:
    python src/metadata/heatmap_field.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, '..', '..', 'results_hyperparam', 'bytetracker', 'bytetracker_optimal')
DATASET_TEST = os.path.join(BASE_DIR, '..', '..', '..', 'SoccerNet', 'tracking', 'test')
OUTPUT_DIR   = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'heatmaps')

SEQUENCE = "SNMOT-118"

# Dimensiones estándar de un campo de fútbol (metros)
FIELD_LENGTH = 105
FIELD_WIDTH  = 68

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar resultados del tracking
result_path = os.path.join(RESULTS_DIR, f"{SEQUENCE}.txt")

if not os.path.exists(result_path):
    raise FileNotFoundError(f"No se encontró {result_path}")

px_x = []
px_y = []

with open(result_path) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

        cx = x + w / 2
        cy = y + h / 2

        px_x.append(cx)
        px_y.append(cy)

px_x = np.array(px_x)
px_y = np.array(px_y)

print(f"Secuencia: {SEQUENCE}")
print(f"Total de posiciones: {len(px_x)}")
print(f"Rango X píxeles: {px_x.min():.0f} - {px_x.max():.0f}")
print(f"Rango Y píxeles: {px_y.min():.0f} - {px_y.max():.0f}")

# ESCALAR POSICIONES A DIMENSIONES DEL CAMPO
x_min, x_max = np.percentile(px_x, 5), np.percentile(px_x, 95)
y_min, y_max = np.percentile(px_y, 5), np.percentile(px_y, 95)

# Recortar valores fuera del rango para que no salgan del campo dibujado
px_x_clipped = np.clip(px_x, x_min, x_max)
px_y_clipped = np.clip(px_y, y_min, y_max)

scale_x = FIELD_LENGTH / (x_max - x_min)
scale_y = FIELD_WIDTH / (y_max - y_min)
scale = min(scale_x, scale_y)

field_x = (px_x_clipped - x_min) * scale
field_y = (y_max - px_y_clipped) * scale

# Centrar verticalmente el contenido dentro del campo
y_range_scaled = (y_max - y_min) * scale
y_offset = (FIELD_WIDTH - y_range_scaled) / 2
field_y = field_y + y_offset

# Dbujar campo de fútbol
def draw_field(ax):
    ax.add_patch(patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH,
                                     facecolor='#3a7d3a', zorder=0))

    line_kwargs = dict(color='white', linewidth=1.5, zorder=3)

    ax.add_patch(patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH,
                                     fill=False, **line_kwargs))

    ax.plot([FIELD_LENGTH/2, FIELD_LENGTH/2], [0, FIELD_WIDTH], **line_kwargs)

    center_circle = patches.Circle((FIELD_LENGTH/2, FIELD_WIDTH/2), 9.15,
                                     fill=False, **line_kwargs)
    ax.add_patch(center_circle)
    ax.plot(FIELD_LENGTH/2, FIELD_WIDTH/2, 'o', color='white', markersize=2, zorder=3)

    area_w, area_h = 16.5, 40.3
    ax.add_patch(patches.Rectangle((0, (FIELD_WIDTH-area_h)/2), area_w, area_h,
                                     fill=False, **line_kwargs))
    ax.add_patch(patches.Rectangle((FIELD_LENGTH-area_w, (FIELD_WIDTH-area_h)/2), area_w, area_h,
                                     fill=False, **line_kwargs))

    small_w, small_h = 5.5, 18.3
    ax.add_patch(patches.Rectangle((0, (FIELD_WIDTH-small_h)/2), small_w, small_h,
                                     fill=False, **line_kwargs))
    ax.add_patch(patches.Rectangle((FIELD_LENGTH-small_w, (FIELD_WIDTH-small_h)/2), small_w, small_h,
                                     fill=False, **line_kwargs))

    goal_w, goal_h = 2, 7.32
    ax.add_patch(patches.Rectangle((-goal_w, (FIELD_WIDTH-goal_h)/2), goal_w, goal_h,
                                     fill=False, **line_kwargs))
    ax.add_patch(patches.Rectangle((FIELD_LENGTH, (FIELD_WIDTH-goal_h)/2), goal_w, goal_h,
                                     fill=False, **line_kwargs))

    ax.set_xlim(-goal_w - 2, FIELD_LENGTH + goal_w + 2)
    ax.set_ylim(-2, FIELD_WIDTH + 2)
    ax.set_aspect('equal')
    ax.axis('off')

# Calcular densidad de posiciones usando KDE
xx, yy = np.mgrid[0:FIELD_LENGTH:200j, 0:FIELD_WIDTH:130j]
positions = np.vstack([field_x, field_y])
kde = gaussian_kde(positions, bw_method=0.15)
density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

# Generar figura
fig, ax = plt.subplots(figsize=(14, 9))

draw_field(ax)

heat = ax.imshow(
    density.T,
    extent=[0, FIELD_LENGTH, 0, FIELD_WIDTH],
    origin='lower',
    cmap='jet',
    alpha=0.6,
    aspect='auto',
    zorder=1
)

ax.set_title(f"Mapa de calor de posiciones — {SEQUENCE}\n"
              f"(BYTETracker, configuración óptima — proyección aproximada, escalado por rango observado)")

cbar = fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Densidad de posiciones (KDE)")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_heatmap_field.png")
plt.savefig(output_path, dpi=150, facecolor='white')
plt.close()

print(f"\nMapa de calor guardado en: {output_path}")