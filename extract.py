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

train_labels = os.listdir(train_path)
print("[INFO] encoding labels...")
le = LabelEncoder() 
le.fit([tl for tl in train_labels])#to find the best match
features = []
labels   = []
i = 0
for label in train_labels:
  cur_path = train_path + "/" + label
  for image_path in glob.glob(cur_path + "/*.png"):
    img = image.load_img(image_path, target_size=(100,100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)# to expand the shape of array
    x = preprocess_input(x)
    print('image shape', x.shape)
    feature = model.predict(x)#Generate the output predictions of input samples
    flat = feature.flatten()#collapse to one dimension
    features.append(flat) #add feature to end of list
    labels.append(label)
    print ("[INFO] processed - {}".format(i))
    i += 1
  print ("[INFO] completed label - {}".format(label))

# encode the labels using LabelEncoder
targetNames = np.unique(labels) #to extract unique labels
le = LabelEncoder()
le_labels = le.fit_transform(labels)
print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels 
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))
h5f_data.close()
h5f_label.close()

print ("[STATUS] features and labels saved..")
print ("[status]end time - {}" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
