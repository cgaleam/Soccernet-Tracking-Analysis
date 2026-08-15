"""
compute_metadata.py — Genera los metadatos deportivos (equipos, mapas de
calor, homografía real con distancia/velocidad, posesión de balón) para un
algoritmo y una secuencia dados.

Reutiliza exactamente la misma lógica ya validada y documentada en
src/metadata/*.py (team_classification.py, heatmap_field.py,
heatmap_by_team.py, heatmap_homography_demo_*.py,
distance_homography_demo_*.py, ball_possesion.py), parametrizada por
algoritmo para poder aplicarse también a la configuración óptima de
OC-SORT y no solo a la de ByteTrack, sin modificar los scripts originales
(ya descritos con detalle exacto en el capítulo de Implementación de la
memoria).

Uso:
    python -m src.web_export.compute_metadata --algo bytetracker
    python -m src.web_export.compute_metadata --algo ocsort
    python -m src.web_export.compute_metadata --algo ocsort --sequence SNMOT-116
"""

import os
import csv
import argparse
from collections import defaultdict, Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

# ─────────────────────────────────────────
# RUTAS Y CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
DATASET_TEST = os.path.join(PROJECT_ROOT, "..", "SoccerNet", "tracking", "test")

RESULTS_DIRS = {
    "bytetracker": os.path.join(PROJECT_ROOT, "results_hyperparam", "bytetracker", "bytetracker_optimal"),
    "ocsort": os.path.join(PROJECT_ROOT, "results_hyperparam", "ocsort", "ocsort_optimal"),
}

SEQUENCES = ["SNMOT-116", "SNMOT-117", "SNMOT-118"]

BALL_MAX_AREA = 2000
BALL_MIN_RATIO = 0.7
BALL_MAX_RATIO = 1.3

FIELD_LENGTH = 105
FIELD_WIDTH = 68

MAX_POSSESSION_DISTANCE_PX = 150
SAMPLE_EVERY = 5
FPS = 25

# Puntos de calibración de homografía por secuencia: dependen únicamente
# de la posición de la cámara en ese tramo, no del algoritmo de tracking,
# así que se reutilizan tal cual (ver sección de homografía en
# Implementación) para proyectar también las posiciones de OC-SORT.
HOMOGRAPHY_CONFIG = {
    "SNMOT-116": {
        "frame_range": (1, 380),
        "excluded_track_ids": [],
        "pixel_points_ref": np.array(
            [[373, 400], [590, 370], [1178, 522], [1408, 484]], dtype=np.float32
        ),
    },
    "SNMOT-118": {
        "frame_range": (350, 650),
        "excluded_track_ids": [328, 274],
        "pixel_points_ref": np.array(
            [[581, 343], [742, 330], [1189, 442], [1369, 414]], dtype=np.float32
        ),
    },
}
REFERENCE_IMG_WIDTH, REFERENCE_IMG_HEIGHT = 1456, 816
ACTUAL_IMG_WIDTH, ACTUAL_IMG_HEIGHT = 1920, 1080
FIELD_POINTS = np.array(
    [[5.5, 24.84], [0.0, 24.84], [5.5, 43.16], [0.0, 43.16]], dtype=np.float32
)


def output_dir(algo, subfolder):
    """metadata_output/<sub>/ para ByteTrack (ya existente, sin tocar);
    metadata_output/ocsort/<sub>/ para OC-SORT (nuevo)."""
    if algo == "bytetracker":
        path = os.path.join(PROJECT_ROOT, "metadata_output", subfolder)
    else:
        path = os.path.join(PROJECT_ROOT, "metadata_output", "ocsort", subfolder)
    os.makedirs(path, exist_ok=True)
    return path


def is_ball(x, y, w, h):
    area = w * h
    ratio = w / h if h > 0 else 1.0
    return area < BALL_MAX_AREA and BALL_MIN_RATIO < ratio < BALL_MAX_RATIO


def load_tracks_by_frame(algo, sequence):
    path = os.path.join(RESULTS_DIRS[algo], f"{sequence}.txt")
    tracks_by_frame = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id, track_id = int(parts[0]), int(parts[1])
            x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
            tracks_by_frame[frame_id].append((track_id, x, y, w, h))
    return tracks_by_frame


def draw_field(ax):
    line_kwargs = dict(color="white", linewidth=1.2, zorder=3)
    ax.add_patch(patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH, facecolor="#3a7d3a", zorder=0))
    ax.add_patch(patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH, fill=False, **line_kwargs))
    ax.plot([FIELD_LENGTH / 2, FIELD_LENGTH / 2], [0, FIELD_WIDTH], **line_kwargs)
    ax.add_patch(patches.Circle((FIELD_LENGTH / 2, FIELD_WIDTH / 2), 9.15, fill=False, **line_kwargs))
    ax.plot(FIELD_LENGTH / 2, FIELD_WIDTH / 2, "o", color="white", markersize=2, zorder=3)
    area_w, area_h = 16.5, 40.3
    ax.add_patch(patches.Rectangle((0, (FIELD_WIDTH - area_h) / 2), area_w, area_h, fill=False, **line_kwargs))
    ax.add_patch(patches.Rectangle((FIELD_LENGTH - area_w, (FIELD_WIDTH - area_h) / 2), area_w, area_h, fill=False, **line_kwargs))
    small_w, small_h = 5.5, 18.3
    ax.add_patch(patches.Rectangle((0, (FIELD_WIDTH - small_h) / 2), small_w, small_h, fill=False, **line_kwargs))
    ax.add_patch(patches.Rectangle((FIELD_LENGTH - small_w, (FIELD_WIDTH - small_h) / 2), small_w, small_h, fill=False, **line_kwargs))
    goal_w, goal_h = 2, 7.32
    ax.add_patch(patches.Rectangle((-goal_w, (FIELD_WIDTH - goal_h) / 2), goal_w, goal_h, fill=False, **line_kwargs))
    ax.add_patch(patches.Rectangle((FIELD_LENGTH, (FIELD_WIDTH - goal_h) / 2), goal_w, goal_h, fill=False, **line_kwargs))
    ax.set_xlim(-goal_w - 2, FIELD_LENGTH + goal_w + 2)
    ax.set_ylim(-2, FIELD_WIDTH + 2)
    ax.set_aspect("equal")
    ax.axis("off")


# ─────────────────────────────────────────
# 1. CLASIFICACIÓN DE EQUIPOS (team_classification.py)
# ─────────────────────────────────────────
def _jersey_color(frame_img, x, y, w, h):
    x, y, w, h = int(x), int(y), int(w), int(h)
    top, bottom = y + int(h * 0.15), y + int(h * 0.50)
    left, right = x + int(w * 0.20), x + int(w * 0.80)
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
    km = KMeans(n_clusters=1, n_init=3, random_state=42)
    km.fit(pixels)
    return km.cluster_centers_[0]


def compute_teams(algo, sequence):
    tracks_by_frame = load_tracks_by_frame(algo, sequence)
    img_folder = os.path.join(DATASET_TEST, sequence, "img1")
    images = sorted(os.listdir(img_folder))
    frame_files = {i + 1: img for i, img in enumerate(images)}

    frame_ids = sorted(tracks_by_frame.keys())
    sampled_frames = frame_ids[::SAMPLE_EVERY]

    detections = []
    for frame_id in sampled_frames:
        img_name = frame_files.get(frame_id)
        if not img_name:
            continue
        frame_img = cv2.imread(os.path.join(img_folder, img_name))
        if frame_img is None:
            continue
        for (track_id, x, y, w, h) in tracks_by_frame[frame_id]:
            if w * h < 500:
                continue
            color = _jersey_color(frame_img, x, y, w, h)
            if color is not None:
                detections.append((frame_id, track_id, color))

    colors = np.array([d[2] for d in detections])
    kmeans_teams = KMeans(n_clusters=2, n_init=10, random_state=42)
    team_labels = kmeans_teams.fit_predict(colors)

    track_teams = defaultdict(list)
    for (frame_id, track_id, _color), team in zip(detections, team_labels):
        track_teams[track_id].append(team)

    final_teams = {
        tid: Counter(teams).most_common(1)[0][0] for tid, teams in track_teams.items()
    }

    out_dir = output_dir(algo, "teams")
    out_path = os.path.join(out_dir, f"{sequence}_teams_final.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "team"])
        for tid, team in sorted(final_teams.items()):
            writer.writerow([tid, team])

    print(f"[{algo}/{sequence}] Equipos: {len(final_teams)} jugadores clasificados -> {out_path}")
    return final_teams


# ─────────────────────────────────────────
# 2. MAPAS DE CALOR (heatmap_field.py, heatmap_by_team.py)
# ─────────────────────────────────────────
def _project_by_range(px_x, px_y):
    x_min, x_max = np.percentile(px_x, 5), np.percentile(px_x, 95)
    y_min, y_max = np.percentile(px_y, 5), np.percentile(px_y, 95)
    px_x_c, px_y_c = np.clip(px_x, x_min, x_max), np.clip(px_y, y_min, y_max)
    scale = min(FIELD_LENGTH / (x_max - x_min), FIELD_WIDTH / (y_max - y_min))
    field_x = (px_x_c - x_min) * scale
    field_y = (y_max - px_y_c) * scale
    y_offset = (FIELD_WIDTH - (y_max - y_min) * scale) / 2
    return field_x, field_y + y_offset


def compute_heatmaps(algo, sequence, track_teams):
    tracks_by_frame = load_tracks_by_frame(algo, sequence)

    px_x, px_y = [], []
    positions_by_team = {0: {"x": [], "y": []}, 1: {"x": [], "y": []}}
    for frame_id, dets in tracks_by_frame.items():
        for (track_id, x, y, w, h) in dets:
            if w * h < 500:
                continue
            cx, cy = x + w / 2, y + h / 2
            px_x.append(cx)
            px_y.append(cy)
            team = track_teams.get(track_id)
            if team is not None:
                positions_by_team[team]["x"].append(cx)
                positions_by_team[team]["y"].append(cy)

    out_dir = output_dir(algo, "heatmaps")

    # -- Mapa general --
    field_x, field_y = _project_by_range(np.array(px_x), np.array(px_y))
    xx, yy = np.mgrid[0:FIELD_LENGTH:200j, 0:FIELD_WIDTH:130j]
    kde = gaussian_kde(np.vstack([field_x, field_y]), bw_method=0.15)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(14, 9))
    draw_field(ax)
    heat = ax.imshow(density.T, extent=[0, FIELD_LENGTH, 0, FIELD_WIDTH], origin="lower",
                      cmap="jet", alpha=0.6, aspect="auto", zorder=1)
    ax.set_title(f"Mapa de calor de posiciones — {sequence} ({algo})\n"
                 f"(proyección aproximada, escalado por rango observado)")
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02).set_label("Densidad de posiciones (KDE)")
    plt.tight_layout()
    general_path = os.path.join(out_dir, f"{sequence}_heatmap_field.png")
    plt.savefig(general_path, dpi=150, facecolor="white")
    plt.close()

    # -- Mapa por equipo --
    all_x = np.array(positions_by_team[0]["x"] + positions_by_team[1]["x"])
    all_y = np.array(positions_by_team[0]["y"] + positions_by_team[1]["y"])
    x_min, x_max = np.percentile(all_x, 5), np.percentile(all_x, 95)
    y_min, y_max = np.percentile(all_y, 5), np.percentile(all_y, 95)
    scale = min(FIELD_LENGTH / (x_max - x_min), FIELD_WIDTH / (y_max - y_min))
    y_offset = (FIELD_WIDTH - (y_max - y_min) * scale) / 2

    def project(px, py):
        px, py = np.clip(px, x_min, x_max), np.clip(py, y_min, y_max)
        return (px - x_min) * scale, (y_max - py) * scale + y_offset

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    team_cmaps = {0: "Blues", 1: "Reds"}
    for team in [0, 1]:
        ax = axes[team]
        tpx = np.array(positions_by_team[team]["x"])
        tpy = np.array(positions_by_team[team]["y"])
        fx, fy = project(tpx, tpy)
        draw_field(ax)
        xx, yy = np.mgrid[0:FIELD_LENGTH:200j, 0:FIELD_WIDTH:130j]
        kde = gaussian_kde(np.vstack([fx, fy]), bw_method=0.15)
        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.imshow(density.T, extent=[0, FIELD_LENGTH, 0, FIELD_WIDTH], origin="lower",
                  cmap=team_cmaps[team], alpha=0.65, aspect="auto", zorder=1)
        ax.set_title(f"Equipo {team} ({len(tpx)} posiciones)")
    fig.suptitle(f"Mapas de calor por equipo — {sequence} ({algo})", fontsize=13)
    plt.tight_layout()
    by_team_path = os.path.join(out_dir, f"{sequence}_heatmap_by_team.png")
    plt.savefig(by_team_path, dpi=150, facecolor="white")
    plt.close()

    print(f"[{algo}/{sequence}] Heatmaps -> {general_path}, {by_team_path}")
    return general_path, by_team_path


# ─────────────────────────────────────────
# 3. HOMOGRAFÍA REAL: HEATMAP + DISTANCIA/VELOCIDAD
#    (heatmap_homography_demo_*.py, distance_homography_demo_*.py)
# ─────────────────────────────────────────
def compute_homography(algo, sequence):
    if sequence not in HOMOGRAPHY_CONFIG:
        return None  # SNMOT-117: sin puntos de calibración (ver Implementación)

    cfg = HOMOGRAPHY_CONFIG[sequence]
    scale_x_img = ACTUAL_IMG_WIDTH / REFERENCE_IMG_WIDTH
    scale_y_img = ACTUAL_IMG_HEIGHT / REFERENCE_IMG_HEIGHT
    pixel_points = cfg["pixel_points_ref"] * [scale_x_img, scale_y_img]
    H, _ = cv2.findHomography(pixel_points, FIELD_POINTS)

    def pixel_to_field(px, py):
        point = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
        t = cv2.perspectiveTransform(point, H)
        return t[0][0][0], t[0][0][1]

    frame_range = cfg["frame_range"]
    tracks_by_frame = load_tracks_by_frame(algo, sequence)
    tracks = defaultdict(list)
    for frame_id, dets in tracks_by_frame.items():
        if not (frame_range[0] <= frame_id <= frame_range[1]):
            continue
        for (track_id, x, y, w, h) in dets:
            if w * h < 500:
                continue
            cx, cy = x + w / 2, y + h  # base del bbox (pies)
            tracks[track_id].append((frame_id, cx, cy))

    for excluded_id in cfg["excluded_track_ids"]:
        tracks.pop(excluded_id, None)

    # -- Distancia y velocidad --
    speed_window = FPS
    results = []
    all_field_x, all_field_y = [], []
    for track_id, points in tracks.items():
        points.sort(key=lambda p: p[0])
        if len(points) < 2:
            continue
        field_positions = [(fid, *pixel_to_field(px, py)) for (fid, px, py) in points]
        for (_, fx, fy) in field_positions:
            all_field_x.append(fx)
            all_field_y.append(fy)

        total_distance = 0.0
        for i in range(1, len(field_positions)):
            f0, x0, y0 = field_positions[i - 1]
            f1, x1, y1 = field_positions[i]
            if f1 - f0 <= 0 or f1 - f0 > 10:
                continue
            total_distance += float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

        speeds_kmh = []
        for i in range(speed_window, len(field_positions)):
            f0, x0, y0 = field_positions[i - speed_window]
            f1, x1, y1 = field_positions[i]
            gap = f1 - f0
            if gap <= 0 or gap > speed_window * 2:
                continue
            dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            speeds_kmh.append((dist / (gap / FPS)) * 3.6)

        results.append({
            "track_id": track_id,
            "num_frames": len(points),
            "distance_m": round(total_distance, 2),
            "avg_speed_kmh": round(float(np.mean(speeds_kmh)), 2) if speeds_kmh else 0,
            "peak_speed_kmh": round(float(np.percentile(speeds_kmh, 95)), 2) if speeds_kmh else 0,
        })
    results.sort(key=lambda r: -r["distance_m"])

    out_dir = output_dir(algo, "distance_homography_demo")
    csv_path = os.path.join(out_dir, f"{sequence}_homography_distance.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "num_frames", "distance_m", "avg_speed_kmh", "peak_speed_kmh"])
        w.writeheader()
        w.writerows(results)

    # -- Heatmap con homografía real --
    heat_dir = output_dir(algo, "heatmap_homography_demo")
    xx, yy = np.mgrid[0:FIELD_LENGTH:200j, 0:FIELD_WIDTH:130j]
    kde = gaussian_kde(np.vstack([all_field_x, all_field_y]), bw_method=0.15)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(14, 9))
    draw_field(ax)
    heat = ax.imshow(density.T, extent=[0, FIELD_LENGTH, 0, FIELD_WIDTH], origin="lower",
                      cmap="jet", alpha=0.65, aspect="auto", zorder=1)
    zone_x_min = max(0, FIELD_POINTS[:, 0].min() - 5)
    zone_x_max = min(FIELD_LENGTH, FIELD_POINTS[:, 0].max() + 15)
    ax.axvspan(0, zone_x_min, color="black", alpha=0.35, zorder=2)
    ax.axvspan(zone_x_max, FIELD_LENGTH, color="black", alpha=0.35, zorder=2)
    ax.set_title(f"Mapa de calor con homografía real — {sequence} [{frame_range[0]}-{frame_range[1]}] ({algo})")
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02).set_label("Densidad de posiciones (KDE)")
    plt.tight_layout()
    heat_path = os.path.join(heat_dir, f"{sequence}_heatmap_homography.png")
    plt.savefig(heat_path, dpi=150, facecolor="white")
    plt.close()

    print(f"[{algo}/{sequence}] Homografía: {len(results)} jugadores -> {csv_path}, {heat_path}")
    return {"csv": csv_path, "heatmap": heat_path, "top5": results[:5]}


# ─────────────────────────────────────────
# 4. POSESIÓN DE BALÓN (ball_possesion.py)
# ─────────────────────────────────────────
def compute_possession(algo, sequence, track_teams):
    det_path = os.path.join(DATASET_TEST, sequence, "det", "det.txt")
    tracks_by_frame = load_tracks_by_frame(algo, sequence)

    players_by_frame = defaultdict(list)
    for frame_id, dets in tracks_by_frame.items():
        for (track_id, x, y, w, h) in dets:
            players_by_frame[frame_id].append((track_id, x + w / 2, y + h / 2))

    ball_by_frame = {}
    with open(det_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
            if is_ball(x, y, w, h):
                ball_by_frame[frame_id] = (x + w / 2, y + h / 2)

    possession_by_frame = {}
    for frame_id, (bx, by) in ball_by_frame.items():
        players = players_by_frame.get(frame_id, [])
        if not players:
            continue
        min_dist, closest = float("inf"), None
        for (track_id, px, py) in players:
            dist = np.sqrt((px - bx) ** 2 + (py - by) ** 2)
            if dist < min_dist:
                min_dist, closest = dist, track_id
        if min_dist > MAX_POSSESSION_DISTANCE_PX:
            continue
        team = track_teams.get(closest)
        if team is not None:
            possession_by_frame[frame_id] = team

    counts = defaultdict(int)
    for team in possession_by_frame.values():
        counts[team] += 1
    total = sum(counts.values())

    out_dir = output_dir(algo, "possession")
    summary_path = os.path.join(out_dir, f"{sequence}_possession_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "frames", "percentage"])
        summary = {}
        for team in sorted(counts.keys()):
            pct = round((counts[team] / total) * 100, 2) if total else 0
            w.writerow([team, counts[team], pct])
            summary[team] = pct

    print(f"[{algo}/{sequence}] Posesión: {dict(counts)} de {total} frames -> {summary_path}")
    return summary


# ─────────────────────────────────────────
# ORQUESTACIÓN
# ─────────────────────────────────────────
def compute_all(algo, sequence):
    print(f"\n=== {algo} / {sequence} ===")
    teams = compute_teams(algo, sequence)
    compute_heatmaps(algo, sequence, teams)
    compute_homography(algo, sequence)
    compute_possession(algo, sequence, teams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["bytetracker", "ocsort"], required=True)
    parser.add_argument("--sequence", choices=SEQUENCES, default=None)
    args = parser.parse_args()

    sequences = [args.sequence] if args.sequence else SEQUENCES
    for seq in sequences:
        compute_all(args.algo, seq)
