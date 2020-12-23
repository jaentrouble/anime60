import tensorflow as tf
from flow_models import *
from model_trainer_half import AnimeModel
from tensorflow.keras import mixed_precision
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
import os
from tools.stitch import *
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-w','--weight', dest='weight',required=True)
parser.add_argument('-n','--name', dest='name',required=True)
args = parser.parse_args()


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

flow_map_size = (512,288)
interp_ratio = [0.4,0.8]
model_f = ehrb0_143_32
weight_dir = args.weight

anime_model = AnimeModel(model_f, interp_ratio, flow_map_size)
anime_model.compile(
    optimizer='adam',
)
anime_model.load_weights(weight_dir)

vid_dir = Path('interp/to_convert')
interp_dir = Path('interp')
vid_names = os.listdir(vid_dir)

for vid_name in vid_names:
    print(f'{vid_name} start')
    cap = cv2.VideoCapture(str(vid_dir/vid_name))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ret, frame = cap.read()
    frame_size = (frame.shape[1],frame.shape[0])
    frame_size_hw = (frame_size[1],frame_size[0])
    writer = cv2.VideoWriter(
        str(interp_dir/f'{os.path.splitext(vid_name)[0]}_interp_{args.name}.mp4'),
        fourcc,
        60,
        frame_size
    )
    f = 0
    # t = tqdm(unit='frames',total=3600*5)

    print('Counting frames...')
    nb_frames = int(subprocess.run(
        [
            'ffprobe', 
            '-v', 
            'fatal', 
            '-count_frames',
            '-select_streams',
            'v:0',
            '-show_entries', 
            'stream=nb_read_frames', 
            '-of', 
            'default=noprint_wrappers=1:nokey=1', 
            str(vid_dir/vid_name)
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT).stdout
    )
    print(f'Total {nb_frames}frames')
    t = tqdm(unit='frames',total=nb_frames)

    while cap.isOpened():
        # f += 1
        # if f > 3600:
        #     break
        if ret:
            frame0 = frame
        else:
            break

        ret, frame = cap.read()
        if ret:
            frame1 = frame
        else:
            break

        ret, frame = cap.read()
        if ret:
            frame2 = frame
        else:
            break

        concated1 = np.concatenate([frame0,frame1],axis=-1).astype(np.float32)/ 255.0
        concated2 = np.concatenate([frame2,frame1],axis=-1).astype(np.float32)/ 255.0
        interped = anime_model.predict_on_batch(np.array([concated1,concated2]))
        interped = np.round(np.clip(interped, 0, 1) * 255).astype(np.uint8)
        interped1, interped2 = interped[0][...,0:3], interped[0][...,3:6]
        interped3, interped4 = interped[1][...,3:6], interped[1][...,0:3]
        writer.write(frame0)
        writer.write(interped1)
        writer.write(interped2)
        writer.write(interped3)
        writer.write(interped4)
        t.update(n=5)

    t.close()
    cap.release()
    writer.release()
    print(f'{vid_name} end')
