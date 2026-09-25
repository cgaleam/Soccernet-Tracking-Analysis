"""
team_visualization.py — Visualiza la clasificación de equipos sobre un frame
del vídeo, dibujando cada jugador con el color de su equipo asignado.

Uso:
    python src/metadata/team_visualization.py
"""

import os
import cv2
import csv

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, '..', '..', 'results_hyperparam', 'bytetracker', 'bytetracker_optimal')
DATASET_TEST = os.path.join(BASE_DIR, '..', '..', '..', 'SoccerNet', 'tracking', 'test')
TEAMS_DIR    = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'teams')
OUTPUT_DIR   = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'teams')

SEQUENCE   = "SNMOT-118"  # secuencia a visualizar
FRAME_TO_SHOW = 100  # frame del vídeo a visualizar

TEAM_COLORS = {
    0: (255, 0, 0),    # azul
    1: (0, 0, 255),    # rojo
}
UNKNOWN_COLOR = (128, 128, 128)  # gris

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar asignación de equipos desde el CSV generado por team_classification.py
teams_csv = os.path.join(TEAMS_DIR, f"{SEQUENCE}_teams_final.csv")

if not os.path.exists(teams_csv):
    raise FileNotFoundError(f"No se encontró {teams_csv}. Ejecuta primero team_classification.py")

track_teams = {}
with open(teams_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        track_teams[int(row['track_id'])] = int(row['team'])

print(f"Equipos cargados para {len(track_teams)} jugadores")

# Cargar tracking elegido
result_path = os.path.join(RESULTS_DIR, f"{SEQUENCE}.txt")

frame_tracks = []
with open(result_path) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        frame_id = int(parts[0])
        if frame_id != FRAME_TO_SHOW:
            continue
        track_id = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        frame_tracks.append((track_id, x, y, w, h))

print(f"Tracks en frame {FRAME_TO_SHOW}: {len(frame_tracks)}")

# Cargar imagen del frame
img_folder = os.path.join(DATASET_TEST, SEQUENCE, "img1")
images = sorted(os.listdir(img_folder))
img_name = images[FRAME_TO_SHOW - 1]  # frame 1-indexed

frame = cv2.imread(os.path.join(img_folder, img_name))

# Dibujar bounding boxes de jugadores con color según equipo
for (track_id, x, y, w, h) in frame_tracks:
    x, y, w, h = int(x), int(y), int(w), int(h)

    team = track_teams.get(track_id)
    color = TEAM_COLORS.get(team, UNKNOWN_COLOR)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    label = f"ID {track_id} | Eq.{team}" if team is not None else f"ID {track_id} | ?"
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Leyenda
cv2.putText(frame, "Equipo 0", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, TEAM_COLORS[0], 3)
cv2.putText(frame, "Equipo 1", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, TEAM_COLORS[1], 3)
cv2.putText(frame, f"{SEQUENCE} | Frame {FRAME_TO_SHOW}", (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# GUARDAR
output_path = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_teams_frame{FRAME_TO_SHOW}.png")
cv2.imwrite(output_path, frame)

print(f"\nVisualización guardada en: {output_path}")