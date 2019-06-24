from keras.models import Sequential
import numpy as np
from keras import backend as K

from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
# def my_init(shape, name=None):
# 	value = np.random.random(shape)
# 	return K.variable(value, name=name)

class MLP_model(object):

	def build():
		model = Sequential()
		model.add(Dense(32, input_dim=17,  kernel_initializer='random_uniform'))
		model.add(BatchNormalization())
		model.add(LeakyReLU(alpha=0.1))
		# model.add(Activation('tanh'))
		# Dropout 1
		model.add(Dropout(0.2))

		model.add(Dense(128, kernel_initializer='random_uniform'))
		model.add(BatchNormalization())
		model.add(LeakyReLU(alpha=0.1))
		model.add(Dropout(0.2))

		model.add(Dense(64, kernel_initializer='random_uniform'))
		model.add(BatchNormalization())
		model.add(Activation('tanh'))
		model.add(Dropout(0.2))

		model.add(Dense(16, kernel_initializer='random_uniform'))
		model.add(BatchNormalization())
		# model.add(LeakyReLU(alpha=0.1))
		model.add(Activation('sigmoid'))
		model.add(Dropout(0.4))

		model.add(Dense(2, kernel_initializer="random_uniform"))
		model.add(BatchNormalization())
		model.add(Activation('softmax'))
		# adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
		#
		# # We add metrics to get more results you want to see
		# model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
		return model