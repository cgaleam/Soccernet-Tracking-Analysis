"""
generate_seqmap_train.py — Genera el archivo seqmap para el split train
Necesario para que trackeval sepa qué secuencias evaluar.
 
Uso:
    python src/evaluation/generate_seqmap_train.py
"""
 
import os
 
# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
DATASET_PATH = "../SoccerNet/tracking/train"
 
seqmaps_dir = os.path.join(DATASET_PATH, "seqmaps")
os.makedirs(seqmaps_dir, exist_ok=True)
 
sequences = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d)) and d.startswith('SNMOT')
])
 
output_file = os.path.join(seqmaps_dir, "SoccerNet-train.txt")
with open(output_file, 'w') as f:
    f.write("name\n")
    for seq in sequences:
        f.write(seq + "\n")
 
print(f"Seqmap train creado con {len(sequences)} secuencias → {output_file}")