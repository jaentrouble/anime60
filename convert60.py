import tensorflow as tf
from flow_models import *
from model_trainer import AnimeModel
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-w','--weight', dest='weight')

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

frame_size = (960,540)
interp_ratio = [0.4,0.8]
model_f = hr_3_2_16
weight_dir = args.weight

inputs = tf.keras.Input((frame_size[1],frame_size[0],6))
anime_model = AnimeModel(inputs, model_f, interp_ratio)
anime_model.compile(
    optimizer='adam',
)
anime_model.load_weights(weight_dir)

vid_dir = Path('interp/to_convert')
vid_paths = [str(vid_dir/vn) for vn in os.listdir(vid_dir)]

for vid_path in vid_paths:
    print(f'{vid_path} start')
    cap = cv2.VideoCapture(vid_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(f'{os.path.splitext(vid_path)[0]}_interp.mp4',fourcc,60,frame_size)
    ret, frame = cap.read()
    t = tqdm(unit='frames')
    while cap.isOpened():
        if ret:
            frame0 = frame
        else:
            break
        # ret, _ = cap.read()
        # if not ret:
        #     break
        
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
        frame0_resized = cv2.resize(frame0, dsize=frame_size)
        frame1_resized = cv2.resize(frame1, dsize=frame_size)
        frame2_resized = cv2.resize(frame2, dsize=frame_size)
        concated1 = np.concatenate([frame0_resized,frame1_resized],axis=-1).astype(np.float32)/ 255.0
        concated2 = np.concatenate([frame2_resized,frame1_resized],axis=-1).astype(np.float32)/ 255.0
        outputs = anime_model(np.array([concated1,concated2]))
        outputs = np.round(np.clip(outputs, 0, 1) * 255).astype(np.uint8)
        interped1, interped2 = outputs[0][...,0:3], outputs[0][...,3:6]
        interped3, interped4 = outputs[1][...,3:6], outputs[1][...,0:3]
        writer.write(frame0_resized)
        writer.write(interped1)
        writer.write(interped2)
        writer.write(interped3)
        writer.write(interped4)
        t.update(n=5)

    t.close()
    cap.release()
    writer.release()
    print(f'{vid_path} end')