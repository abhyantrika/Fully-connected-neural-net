from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import numpy as np 
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD,Adam,RMSprop
import scipy.io as sio

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print X_train.shape,X_test.shape,y_test.shape,y_train.shape
X_train = np.reshape(X_train,(60000,784))
y_train = np.reshape(y_train,(60000,1))
X_test = np.reshape(X_test,(10000,784))
y_test = np.reshape(y_test,(10000,1))


y_train = np_utils.to_categorical(y_train, nb_classes=10)
print y_train.shape,y_train
y_test = np_utils.to_categorical(y_test, nb_classes=10)

model = Sequential()
model.add(Dense(512,input_dim=784,init = 'uniform',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,input_dim = 512,init = 'uniform',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,input_dim = 512,init = 'uniform',activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer = RMSprop(), metrics=['accuracy'])
model.fit(X_train,y_train,nb_epoch = 20,batch_size = 128,validation_data = (X_test,y_test))

scores = model.evaluate(X_test, y_test,verbose=0)
print scores
