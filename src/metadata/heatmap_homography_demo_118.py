"""
heatmap_homography_demo.py — Genera un mapa de calor con homografía real
sobre un campo 2D, para el mismo tramo y configuración usados en el
cálculo de distancia/velocidad.

LIMITACIÓN: la homografía es fiable solo en la zona próxima a los puntos
de calibración. Track_ids que se alejan significativamente de esa zona
(detectados por saltos de posición incompatibles con movimiento humano
real) se excluyen manualmente.

Uso:
    python src/metadata/heatmap_homography_demo.py
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
OUTPUT_DIR   = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'heatmap_homography_demo')

SEQUENCE    = "SNMOT-118"
FRAME_RANGE = (350, 650)
EXCLUDED_TRACK_IDS = [328, 274]
REFERENCE_IMG_WIDTH, REFERENCE_IMG_HEIGHT = 1456, 816
ACTUAL_IMG_WIDTH, ACTUAL_IMG_HEIGHT = 1920, 1080

pixel_points_ref = np.array([
    [581, 343],
    [742, 330],
    [1189, 442],
    [1369, 414],
], dtype=np.float32)

FIELD_LENGTH = 105
FIELD_WIDTH  = 68

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Calcular homografia
scale_x_img = ACTUAL_IMG_WIDTH / REFERENCE_IMG_WIDTH
scale_y_img = ACTUAL_IMG_HEIGHT / REFERENCE_IMG_HEIGHT
pixel_points = pixel_points_ref * [scale_x_img, scale_y_img]

field_points = np.array([
    [5.5, 24.84],
    [0.0, 24.84],
    [5.5, 43.16],
    [0.0, 43.16],
], dtype=np.float32)

H, _ = cv2.findHomography(pixel_points, field_points)

def pixel_to_field(px, py):
    point = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(point, H)
    return transformed[0][0][0], transformed[0][0][1]

# Cargar tracking del tramo
result_path = os.path.join(RESULTS_DIR, f"{SEQUENCE}.txt")

if not os.path.exists(result_path):
    raise FileNotFoundError(f"No se encontró {result_path}")

tracks = {}
with open(result_path) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        frame_id = int(parts[0])
        if not (FRAME_RANGE[0] <= frame_id <= FRAME_RANGE[1]):
            continue
        track_id = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        if w * h < 500:
            continue
        cx = x + w / 2
        cy = y + h  # base del bbox (pies)
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append((frame_id, cx, cy))

for excluded_id in EXCLUDED_TRACK_IDS:
    tracks.pop(excluded_id, None)

print(f"Secuencia: {SEQUENCE} | Tramo: {FRAME_RANGE}")
print(f"Track IDs analizados: {len(tracks)} (excluidos: {EXCLUDED_TRACK_IDS})")

# Proyectar posiciones a metros
field_x, field_y = [], []
for track_id, points in tracks.items():
    for (frame_id, px, py) in points:
        fx, fy = pixel_to_field(px, py)
        field_x.append(fx)
        field_y.append(fy)

field_x = np.array(field_x)
field_y = np.array(field_y)
print(f"Total de posiciones proyectadas: {len(field_x)}")
print(f"Rango X (metros): {field_x.min():.1f} - {field_x.max():.1f}")
print(f"Rango Y (metros): {field_y.min():.1f} - {field_y.max():.1f}")

# Dibujar el campo y el mapa de calor
def draw_field(ax):
    ax.add_patch(patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH,
                                     facecolor='#3a7d3a', zorder=0))
    line_kwargs = dict(color='white', linewidth=1.2, zorder=3)
    ax.add_patch(patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH,
                                     fill=False, **line_kwargs))
    ax.plot([FIELD_LENGTH/2, FIELD_LENGTH/2], [0, FIELD_WIDTH], **line_kwargs)
    ax.add_patch(patches.Circle((FIELD_LENGTH/2, FIELD_WIDTH/2), 9.15,
                                  fill=False, **line_kwargs))
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

# Kde y figura
xx, yy = np.mgrid[0:FIELD_LENGTH:200j, 0:FIELD_WIDTH:130j]
positions = np.vstack([field_x, field_y])
kde = gaussian_kde(positions, bw_method=0.15)
density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(14, 9))
draw_field(ax)

heat = ax.imshow(
    density.T,
    extent=[0, FIELD_LENGTH, 0, FIELD_WIDTH],
    origin='lower',
    cmap='jet',
    alpha=0.65,
    aspect='auto',
    zorder=1
)

# Sombrear la zona NO cubierta por la calibración (fuera de los 4 puntos
# +/- margen), para que sea visualmente honesto sobre dónde es fiable
zone_x_min = max(0, field_points[:, 0].min() - 5)
zone_x_max = min(FIELD_LENGTH, field_points[:, 0].max() + 15)
ax.axvspan(0, zone_x_min, color='black', alpha=0.35, zorder=2)
ax.axvspan(zone_x_max, FIELD_LENGTH, color='black', alpha=0.35, zorder=2)

ax.set_title(f"Mapa de calor con homografía real — {SEQUENCE} [{FRAME_RANGE[0]}-{FRAME_RANGE[1]}]\n"
              f"(BYTETracker óptimo — zona sombreada: fuera del área de calibración fiable)")

cbar = fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Densidad de posiciones (KDE)")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_heatmap_homography.png")
plt.savefig(output_path, dpi=150, facecolor='white')
plt.close()

print(f"\nMapa de calor con homografía guardado en: {output_path}")