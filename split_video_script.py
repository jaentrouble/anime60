import cv2
import numpy as np
import time
from pathlib import Path
import argparse
import ffmpeg
import sys
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-d','--duration', dest='duration')
args = parser.parse_args()

raw_dir = Path('data/raw')
cut_dir = Path('data/cut')
duration = int(args.duration)

start_time = time.time()

for v in os.listdir(raw_dir):
    vid_len = float(subprocess.run(
        [
            'ffprobe', 
            '-v', 
            'error', 
            '-show_entries', 
            'format=duration', 
            '-of', 
            'default=noprint_wrappers=1:nokey=1', 
            str(raw_dir/v)
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT).stdout
    )
    split = int(vid_len // duration)
    for i in range(split):
        try:
            (
                ffmpeg
                .input(str(raw_dir/v),ss=duration*i, t=duration, an='')
                .video
                .output(str(cut_dir/f'{Path(v).stem}_{i}.mp4'), 
                        vcodec='h264_nvenc',
                        video_bitrate='500M',)
                .run()
            )
        except ffmpeg.Error as e:
            print(e.stderr.decode(), file=sys.stderr)
            sys.exit(1)

print(f'took {time.time()-start_time} seconds')