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
