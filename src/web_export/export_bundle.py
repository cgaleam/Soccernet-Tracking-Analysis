"""
export_bundle.py — Genera el paquete de datos estático que consume la
aplicación web (frontend/public/data/): un vídeo anotado recortado por
secuencia y algoritmo, las imágenes de metadatos ya calculadas (por
compute_metadata.py y por el pipeline original de src/metadata/), y dos
ficheros JSON (sequences.json, global_metrics.json) con todo lo demás.

No se ejecuta ningún tracker de nuevo: los vídeos se generan dibujando
directamente los resultados ya calculados (results_hyperparam/*/*_optimal/),
igual que hace el pipeline de metadatos con las posiciones.

Uso:
    python -m src.web_export.export_bundle
"""

import os
import csv
import json
import shutil
from collections import defaultdict

import cv2

from src.web_export.compute_metadata import (
    PROJECT_ROOT, DATASET_TEST, RESULTS_DIRS, SEQUENCES,
    HOMOGRAPHY_CONFIG, is_ball, load_tracks_by_frame, output_dir,
)

FRONTEND_DATA = os.path.join(PROJECT_ROOT, "frontend", "public", "data")
VIDEOS_DIR = os.path.join(FRONTEND_DATA, "videos")
IMAGES_DIR = os.path.join(FRONTEND_DATA, "images")
MEMORIA_FIGURES = os.path.join(PROJECT_ROOT, "memoria", "figures")

ALGOS = ["bytetracker", "ocsort"]
VIDEO_WIDTH = 960

# Tramo de vídeo por secuencia: reutiliza el mismo tramo de la homografía
# donde existe (116, 118); SNMOT-117 no tiene calibración, se usa un
# tramo por defecto de longitud comparable.
VIDEO_RANGES = {
    "SNMOT-116": (1, 380),
    "SNMOT-117": (1, 400),
    "SNMOT-118": (350, 650),
}

TEAM_COLORS_BGR = {0: (255, 120, 0), 1: (0, 0, 255)}  # equipo 0 azul, equipo 1 rojo
UNKNOWN_COLOR_BGR = (150, 150, 150)
BALL_COLOR_BGR = (0, 220, 255)


def load_teams(algo, sequence):
    path = os.path.join(output_dir(algo, "teams"), f"{sequence}_teams_final.csv")
    teams = {}
    if not os.path.exists(path):
        return teams
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            teams[int(row["track_id"])] = int(row["team"])
    return teams


def export_video(algo, sequence):
    frame_range = VIDEO_RANGES[sequence]
    tracks_by_frame = load_tracks_by_frame(algo, sequence)
    teams = load_teams(algo, sequence)

    det_path = os.path.join(DATASET_TEST, sequence, "det", "det.txt")
    ball_by_frame = defaultdict(list)
    with open(det_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            if not (frame_range[0] <= frame_id <= frame_range[1]):
                continue
            x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
            if is_ball(x, y, w, h):
                ball_by_frame[frame_id].append((x, y, w, h))

    img_folder = os.path.join(DATASET_TEST, sequence, "img1")
    images = sorted(os.listdir(img_folder))
    frame_files = {i + 1: img for i, img in enumerate(images)}

    first = cv2.imread(os.path.join(img_folder, frame_files[frame_range[0]]))
    h0, w0 = first.shape[:2]
    scale = VIDEO_WIDTH / w0
    out_w, out_h = VIDEO_WIDTH, int(h0 * scale)

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    out_path = os.path.join(VIDEOS_DIR, f"{sequence}_{algo}.mp4")
    # avc1 (H.264) en vez de mp4v: mp4v (MPEG-4 Part 2) no lo reproducen
    # los navegadores; avc1 sí, y aquí se codifica vía el backend FFmpeg
    # con el que está compilado OpenCV en este entorno.
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), 25.0, (out_w, out_h))

    for frame_id in range(frame_range[0], frame_range[1] + 1):
        img_name = frame_files.get(frame_id)
        if not img_name:
            continue
        frame = cv2.imread(os.path.join(img_folder, img_name))
        if frame is None:
            continue
        frame = cv2.resize(frame, (out_w, out_h))

        for (track_id, x, y, w, h) in tracks_by_frame.get(frame_id, []):
            x1, y1, x2, y2 = [int(v * scale) for v in (x, y, x + w, y + h)]
            color = TEAM_COLORS_BGR.get(teams.get(track_id), UNKNOWN_COLOR_BGR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, str(track_id), (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        for (x, y, w, h) in ball_by_frame.get(frame_id, []):
            x1, y1, x2, y2 = [int(v * scale) for v in (x, y, x + w, y + h)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR_BGR, 2)

        cv2.putText(frame, f"{algo} | {sequence} | frame {frame_id}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        writer.write(frame)

    writer.release()
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[{algo}/{sequence}] Vídeo -> {out_path} ({size_mb:.1f} MB)")
    return f"data/videos/{sequence}_{algo}.mp4"


def copy_image(src, name):
    if not src or not os.path.exists(src):
        return None
    os.makedirs(IMAGES_DIR, exist_ok=True)
    dst = os.path.join(IMAGES_DIR, name)
    shutil.copyfile(src, dst)
    return f"data/images/{name}"


def read_top5_distance(algo, sequence):
    path = os.path.join(output_dir(algo, "distance_homography_demo"), f"{sequence}_homography_distance.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "track_id": int(r["track_id"]),
            "distance_m": float(r["distance_m"]),
            "avg_speed_kmh": float(r["avg_speed_kmh"]),
            "peak_speed_kmh": float(r["peak_speed_kmh"]),
        }
        for r in rows[:5]
    ]


def read_possession(algo, sequence):
    path = os.path.join(output_dir(algo, "possession"), f"{sequence}_possession_summary.csv")
    if not os.path.exists(path):
        return None
    summary = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            summary[row["team"]] = float(row["percentage"])
    return summary


def build_sequences_json():
    entries = []
    for sequence in SEQUENCES:
        for algo in ALGOS:
            teams = load_teams(algo, sequence)
            team_counts = defaultdict(int)
            for t in teams.values():
                team_counts[t] += 1

            heat_dir = output_dir(algo, "heatmaps")
            homog_dir = output_dir(algo, "heatmap_homography_demo")
            has_homography = sequence in HOMOGRAPHY_CONFIG

            entries.append({
                "sequence": sequence,
                "algorithm": algo,
                "playersClassified": len(teams),
                "teamCounts": {"0": team_counts.get(0, 0), "1": team_counts.get(1, 0)},
                "videoUrl": export_video(algo, sequence),
                "heatmapGeneral": copy_image(
                    os.path.join(heat_dir, f"{sequence}_heatmap_field.png"),
                    f"{sequence}_{algo}_heatmap_field.png"),
                "heatmapByTeam": copy_image(
                    os.path.join(heat_dir, f"{sequence}_heatmap_by_team.png"),
                    f"{sequence}_{algo}_heatmap_by_team.png"),
                "heatmapHomography": copy_image(
                    os.path.join(homog_dir, f"{sequence}_heatmap_homography.png"),
                    f"{sequence}_{algo}_heatmap_homography.png") if has_homography else None,
                "topDistanceSpeed": read_top5_distance(algo, sequence),
                "possessionPct": read_possession(algo, sequence),
                "videoFrameRange": list(VIDEO_RANGES[sequence]),
            })
    return entries


# Métricas globales de la comparativa de algoritmos: mismos valores reales
# ya presentados en el capítulo de Validación y resultados de la memoria
# (evaluation_train/, evaluation_test/, evaluation_hyperparam/), no se
# recalculan aquí para no duplicar la fuente de verdad.
GLOBAL_METRICS = {
    "bySplit": {
        "train": {
            "bytetracker": {"HOTA": 73.274, "HOTA0": 82.97, "DetA": 81.585, "AssA": 65.862, "MOTA": 92.146, "IDF1": 77.993, "IDSW": 3866},
            "ocsort":      {"HOTA": 80.227, "HOTA0": 80.338, "DetA": 92.109, "AssA": 69.878, "MOTA": 92.013, "IDF1": 75.343, "IDSW": 2205},
        },
        "test": {
            "bytetracker": {"HOTA": 70.733, "HOTA0": 80.196, "DetA": 80.541, "AssA": 62.182, "MOTA": 90.779, "IDF1": 74.356, "IDSW": 5561},
            "ocsort":      {"HOTA": 75.429, "HOTA0": 75.509, "DetA": 87.233, "AssA": 65.221, "MOTA": 87.013, "IDF1": 71.187, "IDSW": 2046},
        },
    },
    "optimal": {
        "bytetracker": {"params": {"match_thresh": 0.9, "track_buffer": 10, "track_thresh": 0.45}, "HOTA0": 81.427, "MOTA": 91.411, "IDF1": 75.653, "IDSW": 3625},
        "ocsort":      {"params": {"iou_threshold": 0.1, "max_age": 10, "min_hits": 1, "det_thresh": 0.3}, "HOTA0": 80.922, "MOTA": 91.551, "IDF1": 75.926, "IDSW": 2681},
    },
}


def read_hyperparam_summary():
    path = os.path.join(PROJECT_ROOT, "evaluation_hyperparam", "summary_table.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "algo": r["algo"],
            "tracker": r["tracker"],
            "param": r["param"],
            "value": r["value"],
            "HOTA0": float(r["HOTA"]),  # summarize_hyperparam.py ya escribe HOTA(0) en esta columna
            "MOTA": float(r["MOTA"]),
            "IDF1": float(r["IDF1"]),
            "IDSW": int(r["IDSW"]),
        }
        for r in rows
    ]


def build_global_metrics_json():
    data = dict(GLOBAL_METRICS)
    data["hyperparameterStudy"] = read_hyperparam_summary()
    data["hotaVsHota0Chart"] = copy_image(
        os.path.join(MEMORIA_FIGURES, "pruebas_hota_vs_hota0.png"), "hota_vs_hota0.png")
    data["hyperparamChartBytetracker"] = copy_image(
        os.path.join(MEMORIA_FIGURES, "pruebas_hiperparam_bytetracker.png"), "hiperparam_bytetracker.png")
    data["hyperparamChartOcsort"] = copy_image(
        os.path.join(MEMORIA_FIGURES, "pruebas_hiperparam_ocsort.png"), "hiperparam_ocsort.png")
    return data


if __name__ == "__main__":
    os.makedirs(FRONTEND_DATA, exist_ok=True)

    print("=== Generando vídeos + imágenes + sequences.json ===")
    sequences = build_sequences_json()
    with open(os.path.join(FRONTEND_DATA, "sequences.json"), "w") as f:
        json.dump(sequences, f, indent=2, ensure_ascii=False)

    print("\n=== Generando global_metrics.json ===")
    global_metrics = build_global_metrics_json()
    with open(os.path.join(FRONTEND_DATA, "global_metrics.json"), "w") as f:
        json.dump(global_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nListo. Datos exportados en: {FRONTEND_DATA}")
