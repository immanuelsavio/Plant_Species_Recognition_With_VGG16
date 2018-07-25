# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.models import model_from_json
import pickle
#from theano.tensor import blas
#from theano.tensor.nnet import opt

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

#setting the backend
from keras import backend as K

K.set_image_dim_ordering('tf')
with open('conf_xception/flower.json') as a:    

# config variables
  config = json.load(a)
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
features_path = config["features_path"]
labels_path   = config["labels_path"]
test_size   = config["test_size"]
results     = config["results"]