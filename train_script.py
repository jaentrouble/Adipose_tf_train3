import numpy as np
from model_trainer import run_training, create_val_data
import encoder_models
import box_models
import model_lr
import tensorflow as tf
import argparse
import os
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('-em','--encodermodel', dest='emodel')
parser.add_argument('-bm','--boxmodel', dest='bmodel')
parser.add_argument('-lr', dest='lr')
parser.add_argument('-n','--name', dest='name')
parser.add_argument('-e','--epochs', dest='epochs')
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

import imageio as io
import json

img_names = os.listdir('data/done')
img = []
img_name_dict = {}
for idx, name in enumerate(img_names):
    img.append(io.imread('data/done/'+name))
    img_name_dict[name] = idx

json_names = os.listdir('data/save')
data = []
for name in json_names:
    with open('data/save/'+name,'r') as j:
        data.extend(json.load(j))
for datum in data :
    datum['image'] = img_name_dict[datum['image']]

data_train = data[:2600]
data_val = data[2600:2900]
data_test = data[2900:]

encoder_f = getattr(encoder_models, args.emodel)
box_f = getattr(box_models, args.bmodel)
lr_f = getattr(model_lr, args.lr)
name = args.name
batch_size = 6

img_size = (480,320)
box_f = box_f

epochs = int(args.epochs)
steps_per_epoch = len(data_train)//batch_size
mixed_float = args.mixed_float

kwargs = dict(
    encoder_f = encoder_f,
    box_f = box_f,
    lr_f = lr_f,
    name = name,
    epochs = epochs,
    steps_per_epoch=steps_per_epoch,
    mixed_float=mixed_float,
    batch_size=batch_size,
    img=img,
    data=data_train,
    val_data = create_val_data(img, data_val, img_size),
    img_size=img_size,
    notebook= False,
)
run_training(**kwargs)