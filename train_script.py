import numpy as np
from model_trainer import run_training
import adipose_models
import model_lr
import argparse
import tensorflow as tf
import os
import imageio as io
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', dest='model')
parser.add_argument('-lr', dest='lr')
parser.add_argument('-n','--name', dest='name')
parser.add_argument('-e','--epochs', dest='epochs')
parser.add_argument('-b','--batchsize', dest='batch', default=32)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-mg','--memorygrow', dest='mem_growth',
                    action='store_true',default=False)
args = parser.parse_args()

if args.mem_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

data_dir = Path('data')
data_groups = next(os.walk(data_dir))[1]
img = []
data = []
img_name_dict = {}
img_idx = 0
for dg in data_groups[:]:
    img_dir = data_dir/dg/'done'
    img_names = os.listdir(img_dir)
    for name in img_names:
        img_path = str(img_dir/name)
        img.append(io.imread(img_path))
        img_name_dict[img_path] = img_idx
        img_idx += 1

    json_dir = data_dir/dg/'save'
    json_names = os.listdir(json_dir)
    dg_data = []
    for name in json_names[:]:
        with open(str(json_dir/name),'r') as j:
            dg_data.extend(json.load(j))
    for dg_datum in dg_data :
        long_img_name = str(img_dir/dg_datum['image'])
        dg_datum['image'] = img_name_dict[long_img_name]
    data.extend(dg_data)

test_num = len(data) // 10
# To make sure data are chosen randomly between data groups
random.shuffle(data)

data_train = data[:-2*test_num]
data_val = data[-2*test_num:-test_num]
data_test = data[-test_num:]

model_f = getattr(adipose_models, args.model)
lr_f = getattr(model_lr, args.lr)
name = args.name
epochs = int(args.epochs)
mixed_float = args.mixed_float
batch_size = int(args.batch)

kwargs = {}
kwargs['model_f'] = model_f
kwargs['lr_f'] = lr_f
kwargs['name'] = name
kwargs['epochs'] = epochs
kwargs['batch_size'] = batch_size
kwargs['train_data'] = data_train
kwargs['val_data'] = data_val
kwargs['img'] = img
kwargs['img_size'] = (200,200)
kwargs['mixed_float'] = mixed_float
kwargs['notebook'] = False

run_training(**kwargs)