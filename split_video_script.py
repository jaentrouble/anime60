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
    .input(str(raw_dir/(name+ext)),vcodec=rcodec)
    .output('pipe:',format='rawvideo',pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)

# cap = cv2.VideoCapture(str(raw_dir/(name+ext)))
while True:
    # writer = cv2.VideoWriter(str(cut_dir/f'{name}_{idx}{ext}'),fourcc, 60, (3840,2160)) 
    out_process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
        .output(str(cut_dir/f'{name}_{idx}.mp4'), vcodec='h264_nvenc',
        video_bitrate='500M')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for i in range(N):
        # ret, frame = cap.read()
        in_bytes = in_process.stdout.read(width*height*3)
        if in_bytes:
            out_process.stdin.write(in_bytes)
            # writer.write(frame)
            total_frames += 1
        else :
            done = True
            break

    out_process.stdin.close()
    out_process.wait()
    print(f'{idx} done')
    idx += 1
    if done:
        break
in_process.wait()

# cap.release()
print(f'{total_frames} frames took {time.time()-start_time} seconds ')