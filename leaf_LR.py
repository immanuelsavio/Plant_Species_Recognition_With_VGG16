#This file is used to classify the plant species using a classification technique called logistic regression

#Importing required packages
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import cPickle
import h5py
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

# load the user configs
with open('conf_vgg16/flower_lr.json') as f:    
	config = json.load(f)
# config variables
test_size = config["test_size"]
seed = config["seed"]
features_path = config["features_path"]
labels_path = config["labels_path"]
results = config["results"]
classifier_path = config["classifier_path"]
train_path = config["train_path"]
num_classes = config["num_classes"]

# import features and labels
h5f_data = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')
seed= 9
seed_value=np.random.seed(seed)
