import os

import numpy as np
from model_trainer_cyclic import run_training
import flow_models_functional
import model_lr
import argparse
import tensorflow as tf
import imageio as io
import json
from pathlib import Path
import random
from edge_models import edge_models as em

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
parser.add_argument('--load',dest='load',default=None)
parser.add_argument('--edge',dest='edge',default=None)
parser.add_argument('--amodel',dest='amodel',default=None)

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
edge_dir = Path('data/edge')
vid_names = os.listdir(vid_dir)

# To make sure data are chosen randomly between data groups
random.shuffle(vid_names)

test_num = len(vid_names) // 10
train_vid_names = vid_names[:-test_num]
val_vid_names = vid_names[-test_num:]

model_f = getattr(flow_models_functional, args.model)
lr_f = getattr(model_lr, args.lr)
name = args.name
epochs = int(args.epochs)
mixed_float = args.mixed_float
batch_size = int(args.batch)
profile = args.profile
steps_per_epoch = int(args.steps)
if steps_per_epoch <=0:
    steps_per_epoch = len(train_vid_names)*50/batch_size

kwargs = {}
kwargs['model_f'] = model_f
kwargs['lr_f'] = lr_f
kwargs['name'] = name
kwargs['epochs'] = epochs
kwargs['batch_size'] = batch_size
kwargs['steps_per_epoch'] = steps_per_epoch
kwargs['vid_dir'] = vid_dir
kwargs['edge_dir'] = edge_dir
kwargs['train_vid_names'] = train_vid_names
kwargs['val_vid_names'] = val_vid_names
kwargs['frame_size'] = (960,540)
kwargs['flow_map_size'] = (512,288)
kwargs['interpolate_ratios'] = [0.5]
kwargs['patch_size'] = (320,320)
kwargs['overlap'] = 20
kwargs['edge_model_f'] = em.ehrb0_112_12
kwargs['mixed_float'] = mixed_float
kwargs['notebook'] = False
kwargs['profile'] = profile
kwargs['edge_model_path'] = args.edge
kwargs['amodel_path'] = args.amodel
kwargs['load_model_path'] = args.load

run_training(**kwargs)