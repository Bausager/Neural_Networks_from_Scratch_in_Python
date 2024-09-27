
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from tqdm import tqdm

import shutil # Delete non-empty folders


import sys
from zipfile import ZipFile
import os
import urllib
import urllib.request

from zipfile import ZipFile

import copy
import cv2
# Import Neural Network Libary
import NNFS

def download_mnist_dataset():
    URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    FILE = 'fashion_mnist_images.zip'
    FOLDER = 'fashion_mnist_images'

    if not os.path.isfile(FILE):
        print(f'Downloading {URL} and saving as {FILE}...')
        urllib.request.urlretrieve(URL, FILE)


        print('Unzipping images...')
        with ZipFile(FILE) as zip_images:
            zip_images.extractall(FOLDER)

def load_mnist_dataset(dataset, path):

    download_mnist_dataset()

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []
    print("Loading {path} into Numpy...")
    # For each label folder
    for label in tqdm(labels):
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):


    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    # And return all the data

    print('Saving data into Numpy file...')
    np.save('X.npy', X)
    np.save('y.npy', y)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    print('Deleting old files...')
    os.remove('fashion_mnist_images.zip')
    shutil.rmtree('fashion_mnist_images')

    print('Done!')

# Loads a MNIST dataset
def load_dataset():
    X = np.load('X.npy')
    y = np.load('y.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    return X, y, X_test, y_test

# TODO: add your code here..
def MNIST_GetDataSet():
    from sklearn.datasets import fetch_openml
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml(name='mnist_784', return_X_y=True, as_frame=False)
    # Convert to [0;1] via scaling (not always needed)
    X = X / 255.
    #print('Done!')
    return X,y





#%%

# Create dataset
#
#X, y = MNIST_GetDataSet()
#create_data_mnist('fashion_mnist_images')


X_train, y_train, X_test, y_test = load_dataset()

keys = np.array(range(X_train.shape[0]))
np.random.shuffle(keys)

X_train = X_train[keys, :, :]
y_train = y_train[keys]


y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)



X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5


#print(np.shape(X_train[0]))
# # Scale and reshape samples
#X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
#X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

#sys.exit(0)
#print(X_train.shape[0])
#sys.exit(0)
#BS = X_train.shape[0]
BS = 10
#%%
# Instantiate the model
model = NNFS.Model()
# Add layers
model.add(NNFS.Layer_Convolutional([BS, 28, 28], 6, 1))
model.add(NNFS.Activation_Sigmoid())
model.add(NNFS.MaxPool(2, 1))
model.add(NNFS.Layer_Reshape())
model.add(NNFS.Layer_Dense(484, 64))
#model.add(NNFS.Layer_Dense(X_train.shape[1], 64, 
#                            weight_regularizer_L1=1e-5, 
#                            weight_regularizer_L2=1e-5,
#                            bias_regularizer_L1=1e-5, 
#                            bias_regularizer_L2=1e-5))
model.add(NNFS.Layer_Dropout(0.1))
model.add(NNFS.Activation_ReLU())
model.add(NNFS.Layer_Dense(64, 64, 
                            weight_regularizer_L1=1e-5, 
                            weight_regularizer_L2=1e-5,
                            bias_regularizer_L1=1e-5, 
                            bias_regularizer_L2=1e-5))
model.add(NNFS.Layer_Dropout(0.1))
model.add(NNFS.Activation_ReLU())
model.add(NNFS.Layer_Dense(64, 10, 
                            weight_regularizer_L1=1e-5, 
                            weight_regularizer_L2=1e-5,
                            bias_regularizer_L1=1e-5, 
                            bias_regularizer_L2=1e-5))
model.add(NNFS.Activation_Softmax())
# Set loss, optimizer and accuracy objects
model.set(
    loss=NNFS.Loss_CategoricalCrossentropy(),
    optimizer=NNFS.Optimizer_Adam(learning_rate=0.005, decay=1e-3),
    accuracy=NNFS.Accuracy_Categorical()
)
# Finalize the model
model.finalize()
# Train the model

model.train(X_train[:BS*10], y_train[:BS*10], validation_data=(X_test, y_test),
                epochs=5, batch_size=BS, print_every='epoch')


model.save('Full.model')



# #%%
# #while True:
# #    model = NNFS.Model.load('Full.model')
# #    
# #    model.train(X_train, y_train, validation_data=None,
# #                epochs=5, batch_size=128, print_every=None)
# #    
# #    model.save('Full.model')
# #    model.save('Full2.model')
# #    
# #    model.evaluate(X_test, y_test)

    

model2 = NNFS.Model.load('Full.model')

model2.train(X_train[:BS*10], y_train[:BS*10], 
            validation_data=(X_test, y_test),
            epochs=3, 
            batch_size=None, 
            print_every='epoch',
            history='epoch')

# model.save('Full.model')
# model.save('Full2.model')

model2.evaluate(X_test, y_test)

print(model2.history_steps)

plt.figure(1)
plt.plot(model2.history_epochs, model2.history_loss)
plt.title('loss')
plt.grid()


plt.figure(2)
plt.plot(model2.history_epochs, model2.history_accuracy)
plt.title('acc')
plt.grid()

plt.figure(3)
plt.plot(model2.history_epochs, model2.history_data_loss)
plt.title('data_loss')
plt.grid()

plt.figure(4)
plt.plot(model2.history_epochs, model2.history_regularization_loss)
plt.title('reg_loss')
plt.grid()

plt.figure(5)
plt.plot(model2.history_epochs, model2.history_learning_rate)
plt.title('lr')
plt.grid()
plt.show()





























