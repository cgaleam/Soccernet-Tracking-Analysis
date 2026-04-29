import os

# Script para crear el seqmap de SoccerNet-train.txt a partir de las carpetas de secuencias en el dataset.
# Para que TrackEval pueda evaluar correctamente, necesitamos un seqmap que liste todas las secuencias a evaluar.
dataset_path = r"C:\Ingeniería del Software\4\2do Cuatri\TFG\path\to\SoccerNet\tracking\train"
seqmaps_dir  = os.path.join(dataset_path, "seqmaps")
os.makedirs(seqmaps_dir, exist_ok=True)

sequences = sorted([
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('SNMOT')
])

with open(os.path.join(seqmaps_dir, "SoccerNet-train.txt"), 'w') as f:
    f.write("name\n")
    for seq in sequences:
        f.write(seq + "\n")

print(f"Seqmap creado con {len(sequences)} secuencias")