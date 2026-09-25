"""
heatmap_by_team.py — Genera mapas de calor de posiciones separados por equipo,
proyectados sobre un campo de fútbol dibujado en 2D.

Usa la clasificación de equipos generada por team_classification.py.

Uso:
    python src/metadata/heatmap_by_team.py
"""

import os
import csv
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
TEAMS_DIR    = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'teams')
OUTPUT_DIR   = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'heatmaps')

SEQUENCE = "SNMOT-117"

FIELD_LENGTH = 105
FIELD_WIDTH  = 68

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar asignación de equipos
teams_csv = os.path.join(TEAMS_DIR, f"{SEQUENCE}_teams_final.csv")

if not os.path.exists(teams_csv):
    raise FileNotFoundError(f"No se encontró {teams_csv}. Ejecuta primero team_classification.py")

track_teams = {}
with open(teams_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        track_teams[int(row['track_id'])] = int(row['team'])

print(f"Equipos cargados para {len(track_teams)} jugadores")

# Cargar dimensiones del frame
img_folder = os.path.join(DATASET_TEST, SEQUENCE, "img1")
images     = sorted(os.listdir(img_folder))
first_frame = cv2.imread(os.path.join(img_folder, images[0]))
h_img, w_img = first_frame.shape[:2]

# Cargar resultados del tracking
result_path = os.path.join(RESULTS_DIR, f"{SEQUENCE}.txt")

if not os.path.exists(result_path):
    raise FileNotFoundError(f"No se encontró {result_path}")

positions_by_team = {0: {'x': [], 'y': []}, 1: {'x': [], 'y': []}}
unclassified = 0

with open(result_path) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        track_id = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

        # Filtrar balón
        if w * h < 500:
            continue

        team = track_teams.get(track_id)
        if team is None:
            unclassified += 1
            continue

        cx = x + w / 2
        cy = y + h / 2

        positions_by_team[team]['x'].append(cx)
        positions_by_team[team]['y'].append(cy)

print(f"Posiciones equipo 0: {len(positions_by_team[0]['x'])}")
print(f"Posiciones equipo 1: {len(positions_by_team[1]['x'])}")
print(f"Posiciones sin equipo (excluidas): {unclassified}")

# CALCULAR ESCALADO GLOBAL (mismo para ambos equipos)
# Usamos todas las posiciones juntas para que ambos mapas
# compartan la misma escala y sean comparables
all_x = np.array(positions_by_team[0]['x'] + positions_by_team[1]['x'])
all_y = np.array(positions_by_team[0]['y'] + positions_by_team[1]['y'])

x_min, x_max = np.percentile(all_x, 5), np.percentile(all_x, 95)
y_min, y_max = np.percentile(all_y, 5), np.percentile(all_y, 95)

scale_x = FIELD_LENGTH / (x_max - x_min)
scale_y = FIELD_WIDTH / (y_max - y_min)
scale = min(scale_x, scale_y)

y_range_scaled = (y_max - y_min) * scale
y_offset = (FIELD_WIDTH - y_range_scaled) / 2

def project(px_x, px_y):
    px_x = np.clip(px_x, x_min, x_max)
    px_y = np.clip(px_y, y_min, y_max)
    field_x = (px_x - x_min) * scale
    field_y = (y_max - px_y) * scale + y_offset
    return field_x, field_y

# Dibujar campo de fútbol
def draw_field(ax):
    ax.add_patch(patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH,
                                     facecolor='#3a7d3a', zorder=0))

    line_kwargs = dict(color='white', linewidth=1.2, zorder=3)

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

# Generar figura
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

team_cmaps = {0: 'Blues', 1: 'Reds'}

for team in [0, 1]:
    ax = axes[team]

    px_x = np.array(positions_by_team[team]['x'])
    px_y = np.array(positions_by_team[team]['y'])

    field_x, field_y = project(px_x, px_y)

    draw_field(ax)

    xx, yy = np.mgrid[0:FIELD_LENGTH:200j, 0:FIELD_WIDTH:130j]
    positions = np.vstack([field_x, field_y])
    kde = gaussian_kde(positions, bw_method=0.15)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    ax.imshow(
        density.T,
        extent=[0, FIELD_LENGTH, 0, FIELD_WIDTH],
        origin='lower',
        cmap=team_cmaps[team],
        alpha=0.65,
        aspect='auto',
        zorder=1
    )

    ax.set_title(f"Equipo {team} ({len(px_x)} posiciones)")

fig.suptitle(f"Mapas de calor por equipo — {SEQUENCE}\n"
              f"(BYTETracker óptimo — proyección aproximada sin homografía)", fontsize=13)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_heatmap_by_team.png")
plt.savefig(output_path, dpi=150, facecolor='white')
plt.close()

print(f"\nMapas de calor por equipo guardados en: {output_path}")