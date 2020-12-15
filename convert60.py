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

patch_size = (320,320)
overlap = 60
interp_ratio = [0.4,0.8]
model_f = ehrb0_143_32
weight_dir = args.weight

inputs = tf.keras.Input((patch_size[1],patch_size[0],6))
anime_model = AnimeModel(inputs, model_f, interp_ratio)
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
        str(interp_dir/f'{os.path.splitext(vid_name)[0]}_interp.mp4'),
        fourcc,
        60,
        frame_size
    )
    f = 0
    t = tqdm(unit='frames',total=1200*5)
    while cap.isOpened():
        f += 1
        if f > 1200:
            break
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

        concated1 = np.concatenate([frame0,frame1],axis=-1).astype(np.float32)/ 255.0
        concated2 = np.concatenate([frame2,frame1],axis=-1).astype(np.float32)/ 255.0
        patches = frame_to_patch_on_batch(np.array([concated1,concated2]),patch_size,overlap)
        outputs = []
        for i in range(7):
            outputs.append(anime_model(patches[i*8:(i+1)*8]))
        outputs = np.concatenate(outputs,axis=0)
        outputs = np.round(np.clip(outputs, 0, 1) * 255).astype(np.uint8)
        interped = patch_to_frame_on_batch(outputs,frame_size_hw,overlap)
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