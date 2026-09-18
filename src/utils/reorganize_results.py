import os
import shutil

for tracker in ['bytetracker', 'ocsort']:
    tracker_dir = os.path.join('results', tracker)
    for seq_name in os.listdir(tracker_dir):
        seq_dir = os.path.join(tracker_dir, seq_name)
        if os.path.isdir(seq_dir):
            txt_file = os.path.join(seq_dir, seq_name + '.txt')
            if os.path.exists(txt_file):
                shutil.move(txt_file, os.path.join(tracker_dir, seq_name + '.txt'))
                os.rmdir(seq_dir)
                print(f'Movido: {tracker}/{seq_name}.txt')

print('Listo.')