
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import os
import urllib
import urllib.request

import copy

# Import Neural Network Libary
import NNFS



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
X, y = MNIST_GetDataSet()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)



# # Scale and reshape samples
# X_train = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
# X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5


#%%
# Instantiate the model
model = NNFS.Model()
# Add layers
model.add(NNFS.Layer_Dense(X_train.shape[1], 32, weight_regularizer_L1=5e-5, bias_regularizer_L1=5e-5))
model.add(NNFS.Layer_Dropout(0.1))
model.add(NNFS.Activation_ReLU())
model.add(NNFS.Layer_Dense(32, 64, weight_regularizer_L1=5e-5, bias_regularizer_L1=5e-5))
model.add(NNFS.Layer_Dropout(0.1))
model.add(NNFS.Activation_ReLU())
model.add(NNFS.Layer_Dense(64, 10))
model.add(NNFS.Activation_Softmax())
# Set loss, optimizer and accuracy objects
model.set(
    loss=NNFS.Loss_CategoricalCrossentropy(),
    optimizer=NNFS.Optimizer_Adam(learning_rate=0.001, decay=1e-3),
    accuracy=NNFS.Accuracy_Categorical()
)
# Finalize the model
model.finalize()
# Train the model

model.train(X_train, y_train, validation_data=None,
                epochs=5, batch_size=128, print_every=1e9)


# model.save('Full.model')

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

    

# model = NNFS.Model.load('Full.model')

# model.train(X_train, y_train, 
#             validation_data=(X_test, y_test),
#             epochs=10, 
#             batch_size=None, 
#             print_every=1e9,
#             history='epoch')

# model.save('Full.model')
# model.save('Full2.model')

# #model.evaluate(X_test, y_test)

# #print(model.history_steps)

# plt.figure(1)
# plt.plot(model.history_epochs, model.history_loss)
# plt.title('loss')
# plt.grid()


# plt.figure(2)
# plt.plot(model.history_epochs, model.history_accuracy)
# plt.title('acc')
# plt.grid()

# plt.figure(3)
# plt.plot(model.history_epochs, model.history_data_loss)
# plt.title('data_loss')
# plt.grid()

# plt.figure(4)
# plt.plot(model.history_epochs, model.history_regularization_loss)
# plt.title('reg_loss')
# plt.grid()

# plt.figure(5)
# plt.plot(model.history_epochs, model.history_learning_rate)
# plt.title('lr')
# plt.grid()
# plt.show()





























