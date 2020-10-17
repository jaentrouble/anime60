import cv2
import numpy as np
import time
from pathlib import Path
import argparse
import ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', dest='name')
parser.add_argument('-e','--ext', dest='ext')
parser.add_argument('-rc','--readcodec', dest='rcodec')
args = parser.parse_args()

raw_dir = Path('data/raw')
cut_dir = Path('data/cut')
name = args.name
ext = args.ext
rcodec = args.rcodec

start_time = time.time()
n = 0
N = 1000
total_frames = 0
idx = 0
done = False
width, height = 3840, 2160
# fourcc = cv2.VideoWriter_fourcc(*'X264')

in_process = (
    ffmpeg
    .input(str(raw_dir/(name+ext)))
)

cap = cv2.VideoCapture(str(raw_dir/(name+ext)))
while(cap.isOpened()):
    # writer = cv2.VideoWriter(str(cut_dir/f'{name}_{idx}{ext}'),fourcc, 60, (3840,2160)) 
    out_process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}',
                vcodec=rcodec)
        .output(str(cut_dir/f'{name}_{idx}{ext}'), vcodec='h264_nvenc')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for i in range(N):
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                out_process.stdin.write(
                    frame.tobytes()
                )
                # writer.write(frame)
                total_frames += 1
            else :
                done = True
                break
        else :
            done = True
            break
    out_process.stdin.close()
    out_process.wait()
    print(f'{idx} done')
    idx += 1
    if done:
        break

cap.release()
print(f'{total_frames} frames took {time.time()-start_time} seconds ')