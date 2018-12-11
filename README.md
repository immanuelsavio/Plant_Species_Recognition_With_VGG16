# Plant Species Recognition With VGG16

This project deals with identification of different spiecies of plant by processing their leaves. The dataset is a collection of 4 available datasets namely
Folio, Flavia, Swedish and 17Flower.
This dataset also contains one more self made dataset called the Leaf-12.
The Leaf-12 dataset has 12 species of leaves which have been photographed in different cameras, lighting conditions and backgrounds.

### Prerequisites

What things you need to install the software and how to install them<br />
You will be needing Python 2.7 or higher.<br />
The following python packages will be needed<br />

1.Numpy<br />
2.Scipy<br />
3.Scikit-Learn<br />
4.glob<br />
5.OpenCV (cv2)<br />
6.H5py<br />
7.os<br />
8.JSON<br />
9.Datetime<br />
10.Time<br />
11.cPickle<br />
12.Seaborn<br />
13.Tensorflow<br />
14.Keras in tensorflow backend<br />

### Keras in Tensorflow Backend
The above code uses keras package in tensorflow backend. To use it in the backend install tensorflow as er the instructions in the Keras with tensorflow website. Then navigate to your home and in the file .keras/keras.json add the code from the keras.json file in the repository. To use tensorflow it is recommended to use a virtual environment installation. To activate the virtual environment navigate to /home/<user>/tensorflow/venv/bin and then tyoe the command ```source activate```

### Logic

This program uses the pre-trained convolutional neural network model of VGG16 from the Keras package to extract the features. Then the extracted features 
are kept in the output folder in files called features.h5 and labels.h5

Then we use different classifing algorithms like K-nearest neighbour, Bagging, Logistic Regression etc to clssify the testing set. 

### Datasets
The four datasets namely [Folio](https://archive.ics.uci.edu/ml/datasets/Folio) , [Flavia](http://flavia.sourceforge.net/), [17Flower](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/) and [Swedish](http://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/) can be downloaded from the above links
