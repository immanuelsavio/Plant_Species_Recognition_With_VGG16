#This file is to classify the flowers from the features extracted using Bagging Classifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import cPickle
import h5py
import os
import json
import seaborn as sns # to provide attractive statistical graph
import matplotlib.pyplot as plt

# load the user configs
with open('conf_vgg16/flower_bb.json') as f:    
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
seed= 9
seed_value=np.random.seed(seed)
# import features and labels
h5f_data = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')
features_string = h5f_data['dataset_1']
labels_string = h5f_label['dataset_1']

features = np.array(features_string)
labels = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print "[INFO] features shape: {}".format(features.shape)
print "[INFO] labels shape: {}".format(labels.shape)

print "[INFO] training started..."
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed_value)

print "[INFO] splitted train and test data..."
print "[INFO] train data  : {}".format(trainData.shape)
print "[INFO] test data   : {}".format(testData.shape)
print "[INFO] train labels: {}".format(trainLabels.shape)
print "[INFO] test labels : {}".format(testLabels.shape)

# use logistic regression as the model
print("[INFO] creating model...")
model = BaggingClassifier(random_state=seed_value)
model.fit(trainData, trainLabels)#trains the model for fixed number of epochs

# use rank-1 and rank-5 predictions
print("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0
rank_5 = 0

# loop over test data
for (label, features) in zip(testLabels, testData):
	# predict the probability of each class label and
	# take the top-5 class labels
	predictions = model.predict_proba(np.atleast_2d(features))[0]
	predictions = np.argsort(predictions)[::-1][:5]

	# rank-1 prediction increment
	if label == predictions[0]:
		rank_1 += 1

	# rank-5 prediction increment
	if label in predictions:
		rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1)) # to print in 2 decimal places
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData)

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print("[INFO] saving model...")
f = open(classifier_path, "w")
f.write(cPickle.dumps(model))
f.close()

# display the confusion matrix
print ("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm,
            annot=True,#data in each cell
            cmap="Set2")#map data in colourspace
plt.show()
