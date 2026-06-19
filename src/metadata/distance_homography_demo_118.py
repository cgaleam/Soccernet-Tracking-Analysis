"""
distance_homography_demo_118.py — Demuestra el cálculo de distancia y velocidad
real usando una homografía manual, aplicada a un tramo con cámara estable
de SNMOT-118.

Los 4 puntos de referencia corresponden a las esquinas del área pequeña,
con medidas FIFA estándar (5.5m x 18.32m), identificadas manualmente en
un frame representativo del tramo estable.

PRUEBA DE CONCEPTO sobre un tramo concreto (ver también la demo de
SNMOT-116 con jugada de córner). No se aplica al pipeline general por
el movimiento continuo de cámara en el resto de la secuencia.

Uso:
    python src/metadata/distance_homography_demo_118.py
"""

import os
import cv2
import numpy as np
import csv

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, '..', '..', 'results_hyperparam', 'bytetracker', 'bytetracker_optimal')
OUTPUT_DIR   = os.path.join(BASE_DIR, '..', '..', 'metadata_output', 'distance_homography_demo')

SEQUENCE        = "SNMOT-118"
FRAME_RANGE     = (350, 650)  # Tramo jugada ataque con cámara estable 
FPS             = 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# PUNTOS DE REFERENCIA PARA HOMOGRAFÍA
# Imagen de referencia: 1456x816 (reescalar si tu frame es 1920x1080)
REFERENCE_IMG_WIDTH  = 1456
REFERENCE_IMG_HEIGHT = 816
ACTUAL_IMG_WIDTH     = 1920
ACTUAL_IMG_HEIGHT    = 1080

scale_x_img = ACTUAL_IMG_WIDTH / REFERENCE_IMG_WIDTH
scale_y_img = ACTUAL_IMG_HEIGHT / REFERENCE_IMG_HEIGHT

# Puntos en píxeles área pequeña (en 1456x816)
pixel_points_ref = np.array([
    [581, 343],   # A: esquina lejana, lado izquierdo
    [742, 330],   # B: esquina cercana al palo izquierdo
    [1189, 442],  # C: esquina lejana, lado derecho
    [1369, 414],  # D: esquina cercana al palo derecho
], dtype=np.float32)

pixel_points = pixel_points_ref * [scale_x_img, scale_y_img]

# Coordenadas reales del campo (metros), área pequeña FIFA (5.5m x 18.32m)
field_points = np.array([
    [5.5, 24.84],   # A
    [0.0, 24.84],   # B
    [5.5, 43.16],   # C
    [0.0, 43.16],   # D
], dtype=np.float32)

print("Puntos de píxel (resolución real):")
for p in pixel_points:
    print(f"  {p}")

# CALCULAR HOMOGRAFÍA
H, status = cv2.findHomography(pixel_points, field_points)
print(f"\nMatriz de homografía:\n{H}")

def pixel_to_field(px, py):
    point = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(point, H)
    return transformed[0][0][0], transformed[0][0][1]

# VALIDACIÓN
print("\n--- Validación de la homografía ---")
for i, (px, py) in enumerate(pixel_points):
    fx, fy = pixel_to_field(px, py)
    expected = field_points[i]
    error = np.sqrt((fx - expected[0])**2 + (fy - expected[1])**2)
    print(f"  Punto {i}: proyectado=({fx:.2f}, {fy:.2f})  esperado=({expected[0]:.2f}, {expected[1]:.2f})  error={error:.3f}m")

# CARGAR TRACKING DEL TRAMO
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

print(f"\nTrack IDs en el tramo [{FRAME_RANGE[0]}-{FRAME_RANGE[1]}]: {len(tracks)}")


# ─────────────────────────────────────────
# EXCLUSIÓN MANUAL DE TRACK_IDS ANÓMALOS
# Tras inspección manual (ver memoria, sección de metodología), estos
# track_ids muestran saltos de posición incompatibles con un movimiento
# humano real al proyectarlos con la homografía (p.ej. saltos de 1px en
# imagen equivalentes a >0.7m en el campo), lo que indica error de
# tracking en zonas alejadas de los puntos de calibración, no un fallo
# de la homografía en sí.
# ─────────────────────────────────────────
EXCLUDED_TRACK_IDS = [328, 274]
 
for excluded_id in EXCLUDED_TRACK_IDS:
    if excluded_id in tracks:
        del tracks[excluded_id]
 
print(f"Track IDs excluidos manualmente: {EXCLUDED_TRACK_IDS}")
print(f"Track IDs analizados: {len(tracks)}")

# CALCULAR DISTANCIA Y VELOCIDAD
SPEED_WINDOW = FPS

results = []

for track_id, points in tracks.items():
    points.sort(key=lambda p: p[0])
    if len(points) < 2:
        continue

    field_positions = []
    for (frame_id, px, py) in points:
        fx, fy = pixel_to_field(px, py)
        field_positions.append((frame_id, fx, fy))

    total_distance = 0.0
    for i in range(1, len(field_positions)):
        f0, x0, y0 = field_positions[i-1]
        f1, x1, y1 = field_positions[i]
        if f1 - f0 <= 0 or f1 - f0 > 10:
            continue
        total_distance += np.sqrt((x1-x0)**2 + (y1-y0)**2)

    speeds_kmh = []
    for i in range(SPEED_WINDOW, len(field_positions)):
        f0, x0, y0 = field_positions[i - SPEED_WINDOW]
        f1, x1, y1 = field_positions[i]
        gap = f1 - f0
        if gap <= 0 or gap > SPEED_WINDOW * 2:
            continue
        dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        speed_kmh = (dist / (gap / FPS)) * 3.6
        speeds_kmh.append(speed_kmh)

    peak_speed = np.percentile(speeds_kmh, 95) if speeds_kmh else 0
    avg_speed  = np.mean(speeds_kmh) if speeds_kmh else 0

    results.append({
        'track_id'       : track_id,
        'num_frames'     : len(points),
        'distance_m'     : round(total_distance, 2),
        'avg_speed_kmh'  : round(avg_speed, 2),
        'peak_speed_kmh' : round(peak_speed, 2),
    })

results.sort(key=lambda r: -r['distance_m'])

# GUARDAR Y MOSTRAR
output_csv = os.path.join(OUTPUT_DIR, f"{SEQUENCE}_homography_distance.csv")
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['track_id', 'num_frames', 'distance_m', 'avg_speed_kmh', 'peak_speed_kmh'])
    writer.writeheader()
    writer.writerows(results)

print(f"\nResultados guardados en: {output_csv}")
print(f"\nDistancia y velocidad — tramo [{FRAME_RANGE[0]}-{FRAME_RANGE[1]}] ({(FRAME_RANGE[1]-FRAME_RANGE[0])/FPS:.1f}s):")
for r in results[:10]:
    print(f"  ID {r['track_id']}: {r['distance_m']}m en {r['num_frames']} frames "
          f"(vel. media: {r['avg_speed_kmh']} km/h, pico p95: {r['peak_speed_kmh']} km/h)")
    

