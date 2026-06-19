"""
ball_possession.py — Calcula la posesión del balón por equipo a partir
de la proximidad espacial entre el balón detectado y los jugadores.

METODOLOGÍA: para cada frame con balón detectado, se calcula qué jugador
(track_id) está más cerca del balón y se asigna la posesión de ese frame
al equipo de dicho jugador.

LIMITACIÓN IMPORTANTE: esto NO es posesión real en sentido futbolístico
(control efectivo del balón con el pie), sino una aproximación por
proximidad espacial entre el centro del balón y el centro del jugador
más cercano. Es una métrica habitual en proyectos de análisis automático
cuando no se dispone de información de contacto/control, pero debe
interpretarse como tal.

Uso:
    python src/metadata/ball_possession.py
"""

import os
import csv
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, '..', '..', 'results_hyperparam', 'bytetracker', 'bytetracker_optimal')
TEAMS_DIR    = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'teams')
OUTPUT_DIR   = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'possession')

SEQUENCE = "SNMOT-118"

# Mismos parámetros de detección de balón usados en el resto del proyecto
BALL_MAX_AREA  = 2000
BALL_MIN_RATIO = 0.7
BALL_MAX_RATIO = 1.3

# Distancia máxima (píxeles) para considerar que un jugador "tiene" el
# balón cerca; evita asignar posesión cuando el jugador más próximo está
# en realidad lejos (balón suelto, en el aire, etc.)
MAX_POSSESSION_DISTANCE_PX = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# CARGAR ASIGNACIÓN DE EQUIPOS
# ─────────────────────────────────────────
teams_csv = os.path.join(TEAMS_DIR, f"{SEQUENCE}_teams_final.csv")

if not os.path.exists(teams_csv):
    raise FileNotFoundError(f"No se encontró {teams_csv}. Ejecuta primero team_classification.py")

track_teams = {}
with open(teams_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        team_value = row['team']
        team = team_value if team_value == 'referee' else int(team_value)
        track_teams[int(row['track_id'])] = team

print(f"Equipos cargados para {len(track_teams)} jugadores")

# ─────────────────────────────────────────
# CARGAR JUGADORES DEL TRACKING Y BALÓN DEL det.txt ORIGINAL
# El balón se excluye antes de pasarse al tracker (ver run_all_test.py /
# bytetracker_optimal.py: is_ball() filtra el balón de player_dets antes
# de llamar a tracker.update()). Por ello el archivo de resultados de
# BYTETracker NUNCA contiene detecciones de balón, y hay que leerlas del
# det.txt original (detecciones crudas, sin tracking ni IDs).
# ─────────────────────────────────────────
DATASET_TEST = os.path.join(BASE_DIR, '..', '..', '..', 'SoccerNet', 'tracking', 'test')
det_path = os.path.join(DATASET_TEST, SEQUENCE, "det", "det.txt")

if not os.path.exists(det_path):
    raise FileNotFoundError(f"No se encontró {det_path}")

# Jugadores: desde el archivo de resultados de tracking (tiene track_id estable)
result_path = os.path.join(RESULTS_DIR, f"{SEQUENCE}.txt")
if not os.path.exists(result_path):
    raise FileNotFoundError(f"No se encontró {result_path}")

players_by_frame = defaultdict(list)
with open(result_path) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        frame_id = int(parts[0])
        track_id = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        cx = x + w / 2
        cy = y + h / 2
        players_by_frame[frame_id].append((track_id, cx, cy))

# Balón: desde el det.txt original (detecciones crudas, sin IDs de tracking)
ball_by_frame = {}
with open(det_path) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        frame_id = int(parts[0])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

        area  = w * h
        ratio = w / h if h > 0 else 1.0

        if area < BALL_MAX_AREA and BALL_MIN_RATIO < ratio < BALL_MAX_RATIO:
            cx = x + w / 2
            cy = y + h / 2
            ball_by_frame[frame_id] = (cx, cy)

print(f"Frames con balón detectado: {len(ball_by_frame)}")
print(f"Frames con jugadores: {len(players_by_frame)}")

# ─────────────────────────────────────────
# CALCULAR JUGADOR MÁS CERCANO AL BALÓN, POR FRAME
# ─────────────────────────────────────────
possession_by_frame = {}  # frame -> team (0, 1, 'referee' o None)

for frame_id, (bx, by) in ball_by_frame.items():
    players = players_by_frame.get(frame_id, [])
    if not players:
        continue

    min_dist = float('inf')
    closest_track_id = None

    for (track_id, px, py) in players:
        dist = np.sqrt((px - bx)**2 + (py - by)**2)
        if dist < min_dist:
            min_dist = dist
            closest_track_id = track_id

    if min_dist > MAX_POSSESSION_DISTANCE_PX:
        continue  # balón suelto / lejos de cualquier jugador, no se asigna

    team = track_teams.get(closest_track_id)
    if team is not None:
        possession_by_frame[frame_id] = team

print(f"Frames con posesión asignada: {len(possession_by_frame)}")

# ─────────────────────────────────────────
# CALCULAR PORCENTAJES DE POSESIÓN
# ─────────────────────────────────────────
counts = defaultdict(int)
for team in possession_by_frame.values():
    counts[team] += 1

total = sum(counts.values())

print(f"\n--- Posesión del balón (por proximidad) ---")
for team in sorted(counts.keys(), key=str):
    pct = (counts[team] / total) * 100 if total > 0 else 0
    label = "Árbitro" if team == 'referee' else f"Equipo {team}"
    print(f"  {label}: {counts[team]} frames ({pct:.1f}%)")

# ─────────────────────────────────────────
# GUARDAR RESULTADOS
# ─────────────────────────────────────────
output_csv = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_possession.csv")
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'team'])
    for frame_id, team in sorted(possession_by_frame.items()):
        writer.writerow([frame_id, team])

output_summary = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_possession_summary.csv")
with open(output_summary, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['team', 'frames', 'percentage'])
    for team in sorted(counts.keys(), key=str):
        pct = (counts[team] / total) * 100 if total > 0 else 0
        writer.writerow([team, counts[team], round(pct, 2)])

print(f"\nResultados guardados en:")
print(f"  {output_csv}")
print(f"  {output_summary}")