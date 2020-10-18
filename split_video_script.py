import cv2
import numpy as np
import time
from pathlib import Path
import argparse
import ffmpeg
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', dest='name')
parser.add_argument('-e','--ext', dest='ext')
# parser.add_argument('-rc','--readcodec', dest='rcodec')
parser.add_argument('-d','--duration', dest='duration')
parser.add_argument('-s','--split', dest='split')
args = parser.parse_args()

raw_dir = Path('data/raw')
cut_dir = Path('data/cut')
name = args.name
ext = args.ext
duration = int(args.duration)
split = int(args.split)

start_time = time.time()
# width, height = 3840, 2160
# fourcc = cv2.VideoWriter_fourcc(*'X264')

for i in range(split):
    try:
        (
            ffmpeg
            .input(str(raw_dir/(name+ext)),ss=duration*i, t=duration)
            .output(str(cut_dir/f'{name}_{i}.mp4'), vcodec='h264_nvenc',
                    video_bitrate='500M')
            .run()
        )
    except ffmpeg.Error as e:
        print(e.stderr.decode(), file=sys.stderr)
        sys.exit(1)

print(f'took {time.time()-start_time} seconds')