
# coding: utf-8

# In[1]:

import numpy as np
#import matplotlib.pyplot as plt


# In[5]:

from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD

# In[3]:

X_train = np.load("X_train.npy")
print X_train.shape

# In[31]:
X_test = np.load("X_test.npy")
print X_test.shape


# In[13]:

nb_classes = 7
y_train = np.load("y_train.npy")
print y_train.shape
y_test = np.load("y_test.npy")
print y_test.shape


# In[4]:

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
X_train = X_train.reshape(294, 150*150)
X_test = X_test.reshape(33, 150*150)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


# In[17]:

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

nb_epochs = 7303
# load json and create model
json_file = open('basic_CNN_CKplus_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# In[6]:

model = loaded_model
csv_logger = CSVLogger('basic_cnn_training_'+str(nb_epochs)+'_log', separator=',', append=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# In[8]:

model.fit(X_train, Y_train, callbacks=[csv_logger, reduce_lr], batch_size=32, nb_epoch=nb_epochs, verbose=1, validation_data=(X_test, Y_test))
model.save_weights("basic_cnn_"+str(nb_epochs)+"_epochs.h5")
print("\nSaved model weights to disk")
