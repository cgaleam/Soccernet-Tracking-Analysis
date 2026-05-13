"""
generate_seqmap_test.py — Genera el archivo seqmap para el split test
Necesario para que trackeval sepa qué secuencias evaluar.

Uso:
    python src/evaluation/generate_seqmap_test.py
"""

import os

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
DATASET_PATH = r"../path/to/SoccerNet/tracking/test"

seqmaps_dir = os.path.join(DATASET_PATH, "seqmaps")
os.makedirs(seqmaps_dir, exist_ok=True)

sequences = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d)) and d.startswith('SNMOT')
])

output_file = os.path.join(seqmaps_dir, "SoccerNet-test.txt")
with open(output_file, 'w') as f:
    f.write("name\n")
    for seq in sequences:
        f.write(seq + "\n")

print(f"Seqmap test creado con {len(sequences)} secuencias → {output_file}")