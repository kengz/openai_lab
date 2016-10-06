# just a sample note, not actually runnable
import numpy as np
np.random.seed(42)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.regularizers import l1, l2

m = Sequential()
m.add(Dense(2048, input_shape=(10,), activation='relu', init='lecun_uniform', W_regularizer=l2(0.01)))
m.add(Dense(1024, activation='relu', init='lecun_uniform', W_regularizer=l2(0.01)))
m.add(Dense(512, activation='relu', init='lecun_uniform', W_regularizer=l2(0.01)))
m.add(Dense(256, activation='relu', init='lecun_uniform', W_regularizer=l2(0.01)))
m.add(Dense(128, activation='relu', init='lecun_uniform', W_regularizer=l2(0.01)))
m.add(Dense(2, activation='softmax', init='lecun_uniform', W_regularizer=l2(0.01)))
m.summary()
sgd = SGD(lr=0.00001, momentum=0.9, nesterov=False)
m.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
m.fit(X, Y, batch_size=128, nb_epoch=20, validation_split=0.2, verbose=2)

# activations = softmax, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear
# init = uniform, lecun_uniform, normal, ...
# optimizers:
# SGD: lr, momentum, decay, nesterov (momentum)
# RMSprop: lr
# Adagrad, Adadelta
# Adam: lr
# Nadam
# 
# loss = mse, mae, mape, msle, squared_hinge, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy, kld, posson, cosine_proximity
# 
# saving the architecture
# from models import model_from_json
# json_string = model.to_json()
# model = model_from_json(json_string)
# 
# model.save_weights(filepath)
# model.load_weights(filepath, by_name=False)
