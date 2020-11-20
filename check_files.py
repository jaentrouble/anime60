import os
import cv2
from pathlib import Path

cut_dir = Path('data/cut')

for v in os.listdir(cut_dir).sort():
    print(f'checking {v}')
    cap = cv2.VideoCapture(str(cut_dir/v))
    while cap.isOpened():
        ret, _ = cap.read()
        if not ret:
            break
    cap.release()
print('finished')