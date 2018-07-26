# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
# keras imports
from keras.layers.core import Dense, Activation, Dropout
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from keras import *

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time
from keras import backend as K
K.set_image_dim_ordering('tf')

# load the user configs
with open('conf_densenet/flower.json') as f:    
  config = json.load(f)
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
features_path = config["features_path"]
labels_path   = config["labels_path"]
test_size   = config["test_size"]
results     = config["results"]
print ("[status]start time - {}" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))

base_model = DenseNet121(include_top=include_top,weights=weights)
model = Model(input=base_model.input, output=base_model.output)
model.summary()

