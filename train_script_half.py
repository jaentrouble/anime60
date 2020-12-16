import os

import numpy as np
from model_trainer_half import run_training
import flow_models
import model_lr
import argparse
import tensorflow as tf
import imageio as io
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', dest='model')
parser.add_argument('-lr', dest='lr')
parser.add_argument('-n','--name', dest='name')
parser.add_argument('-e','--epochs', dest='epochs')
parser.add_argument('-s','--steps', dest='steps', default=0)
parser.add_argument('-b','--batchsize', dest='batch', default=32)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-mg','--memorygrow', dest='mem_growth',
                    action='store_true',default=False)
parser.add_argument('-pf','--profile', dest='profile',
                    action='store_true',default=False)
parser.add_argument('--load',dest='load',default=False)

args = parser.parse_args()

if args.mem_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

vid_dir = Path('data/cut')
vid_paths = [str(vid_dir/vn) for vn in os.listdir(vid_dir)]

# To make sure data are chosen randomly between data groups
random.shuffle(vid_paths)

test_num = len(vid_paths) // 10
train_vid_paths = vid_paths[:-2*test_num]
val_vid_paths = vid_paths[-2*test_num:-test_num]
test_vid_paths = vid_paths[-test_num:]

model_f = getattr(flow_models, args.model)
lr_f = getattr(model_lr, args.lr)
name = args.name
epochs = int(args.epochs)
mixed_float = args.mixed_float
batch_size = int(args.batch)
profile = args.profile
steps_per_epoch = int(args.steps)
if steps_per_epoch <=0:
    steps_per_epoch = len(train_vid_paths)*50/batch_size
load_model_path = args.load

kwargs = {}
kwargs['model_f'] = model_f
kwargs['lr_f'] = lr_f
kwargs['name'] = name
kwargs['epochs'] = epochs
kwargs['batch_size'] = batch_size
kwargs['steps_per_epoch'] = steps_per_epoch
kwargs['train_vid_paths'] = train_vid_paths
kwargs['val_vid_paths'] = val_vid_paths
kwargs['test_vid_paths'] = val_vid_paths
kwargs['frame_size'] = (960,540)
kwargs['flow_map_size'] = (640,352)
kwargs['interpolate_ratios'] = [0.5]
kwargs['mixed_float'] = mixed_float
kwargs['notebook'] = False
kwargs['profile'] = profile
kwargs['load_model_path'] = load_model_path

run_training(**kwargs)