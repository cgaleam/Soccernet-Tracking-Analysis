import os
import shutil

for tracker in ['bytetracker', 'ocsort']:
    tracker_dir = f'results/{tracker}'
    for filename in os.listdir(tracker_dir):
        if filename.endswith('.txt'):
            seq_name = filename.replace('.txt', '')
            seq_dir = os.path.join(tracker_dir, seq_name)
            os.makedirs(seq_dir, exist_ok=True)
            shutil.move(
                os.path.join(tracker_dir, filename),
                os.path.join(seq_dir, filename)
            )
            print(f'Movido: {tracker}/{filename} → {tracker}/{seq_name}/{filename}')

print('Reorganización completada.')