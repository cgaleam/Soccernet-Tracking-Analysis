"""
team_classification.py — Clasifica a los jugadores en dos equipos
a partir del color dominante de su camiseta, usando K-means clustering.

Para cada bounding box del tracking:
1. Se recorta la región superior del jugador (camiseta)
2. Se extrae el color dominante con K-means (k=1) sobre los píxeles
3. Se agrupan todos los colores dominantes de todos los jugadores en
   2 clusters (k=2) usando K-means, asumiendo 2 equipos

Resultado: un CSV con frame, track_id, team (0 o 1) por cada detección.

Uso:
    python src/metadata/team_classification.py
"""

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, '..', '..', 'results_hyperparam', 'bytetracker', 'bytetracker_optimal')
DATASET_TEST = os.path.join(BASE_DIR, '..', '..', '..', 'SoccerNet', 'tracking', 'test')
OUTPUT_DIR   = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'teams')

SEQUENCE = "SNMOT-118"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CARGAR RESULTADOS DE TRACKING
result_path = os.path.join(RESULTS_DIR, f"{SEQUENCE}.txt")

if not os.path.exists(result_path):
    raise FileNotFoundError(f"No se encontró {result_path}")

tracks_by_frame = {}

with open(result_path) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        frame_id  = int(parts[0])
        track_id  = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

        if frame_id not in tracks_by_frame:
            tracks_by_frame[frame_id] = []
        tracks_by_frame[frame_id].append((track_id, x, y, w, h))

print(f"Secuencia: {SEQUENCE}")
print(f"Frames con tracks: {len(tracks_by_frame)}")

# EXTRAER COLOR DOMINANTE DE CAMISETA POR DETECCIÓN
img_folder = os.path.join(DATASET_TEST, SEQUENCE, "img1")

def get_jersey_color(frame_img, x, y, w, h):
    """Extrae el color dominante de la parte superior del bounding box (camiseta)."""
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Recortar la zona del torso: parte superior-central del bbox
    top    = y + int(h * 0.15)
    bottom = y + int(h * 0.50)
    left   = x + int(w * 0.20)
    right  = x + int(w * 0.80)

    # Asegurar límites válidos
    top, bottom = max(0, top), min(frame_img.shape[0], bottom)
    left, right = max(0, left), min(frame_img.shape[1], right)

    if bottom <= top or right <= left:
        return None

    crop = frame_img[top:bottom, left:right]
    if crop.size == 0:
        return None

    pixels = crop.reshape(-1, 3).astype(np.float64)

    if len(pixels) < 5:
        return None

    # K-means con k=1 para obtener el color dominante (centroide)
    kmeans = KMeans(n_clusters=1, n_init=3, random_state=42)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]  # BGR

    return dominant_color

# PROCESAR FRAMES (muestreo cada N frames para velocidad)
SAMPLE_EVERY = 5  # procesar 1 de cada 5 frames para acelerar

detections = []  # (frame_id, track_id, color_bgr)

images = sorted(os.listdir(img_folder))
frame_files = {i+1: img for i, img in enumerate(images)}  # frame 1-indexed

frame_ids = sorted(tracks_by_frame.keys())
sampled_frames = frame_ids[::SAMPLE_EVERY]

print(f"Procesando {len(sampled_frames)} de {len(frame_ids)} frames (muestreo cada {SAMPLE_EVERY})...")

for frame_id in sampled_frames:
    img_name = frame_files.get(frame_id)
    if not img_name:
        continue

    frame_img = cv2.imread(os.path.join(img_folder, img_name))
    if frame_img is None:
        continue

    for (track_id, x, y, w, h) in tracks_by_frame[frame_id]:
        # Filtrar el balón (muy pequeño)
        if w * h < 500:
            continue

        color = get_jersey_color(frame_img, x, y, w, h)
        if color is not None:
            detections.append((frame_id, track_id, color))

print(f"Detecciones procesadas: {len(detections)}")

# AGRUPAR EN 2 EQUIPOS CON K-MEANS (k=2)
colors = np.array([d[2] for d in detections])

kmeans_teams = KMeans(n_clusters=2, n_init=10, random_state=42)
team_labels = kmeans_teams.fit_predict(colors)

print(f"\nCentroides de color por equipo (BGR):")
for i, center in enumerate(kmeans_teams.cluster_centers_):
    count = np.sum(team_labels == i)
    print(f"  Equipo {i}: BGR={center.astype(int)} — {count} detecciones")

# GUARDAR RESULTADOS
output_csv = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_teams.csv")

with open(output_csv, 'w') as f:
    f.write("frame,track_id,team,b,g,r\n")
    for (frame_id, track_id, color), team in zip(detections, team_labels):
        b, g, r = color.astype(int)
        f.write(f"{frame_id},{track_id},{team},{b},{g},{r}\n")

print(f"\nResultados guardados en: {output_csv}")

# DETERMINAR EQUIPO MAYORITARIO POR track_id
# (un jugador puede tener varias muestras con team distinto por ruido)
from collections import defaultdict, Counter

track_teams = defaultdict(list)
for (frame_id, track_id, color), team in zip(detections, team_labels):
    track_teams[track_id].append(team)

print(f"\nAsignación final por track_id ({len(track_teams)} jugadores):")
final_teams = {}
for track_id, teams in sorted(track_teams.items()):
    most_common = Counter(teams).most_common(1)[0][0]
    final_teams[track_id] = most_common
    print(f"  ID {track_id}: Equipo {most_common} ({len(teams)} muestras, {Counter(teams)})")

# Guardar asignación final
output_final = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_teams_final.csv")
with open(output_final, 'w') as f:
    f.write("track_id,team\n")
    for track_id, team in sorted(final_teams.items()):
        f.write(f"{track_id},{team}\n")

print(f"\nAsignación final guardada en: {output_final}")