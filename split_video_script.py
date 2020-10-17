import cv2
import numpy as np
import time
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', dest='name')
parser.add_argument('-e','--ext', dest='ext')
args = parser.parse_args()
raw_dir = Path('data/raw')
cut_dir = Path('data/cut')
name = 'ryza4k60_1'
ext = '.mp4'

start_time = time.time()
n = 0
N = 1000
total_frames = 0
idx = 0
done = False
fourcc = cv2.VideoWriter_fourcc(*'X264')
cap = cv2.VideoCapture(str(raw_dir/(name+ext)))
while(cap.isOpened()):
    writer = cv2.VideoWriter(str(cut_dir/f'{name}_{idx}{ext}'),fourcc, 60, (3840,2160)) 
    for i in range(N):
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
                total_frames += 1
            else :
                done = True
                break
        else :
            done = True
            break
    writer.release()
    print(f'{idx} done')
    idx += 1
    if done:
        break

cap.release()
print(f'{total_frames} frames took {time.time()-start_time} seconds ')