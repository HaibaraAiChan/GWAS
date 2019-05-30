import matplotlib.pyplot as plt
import pandas as pd

import keras

from keras.layers import Dense, Dropout, Activation

from keras.callbacks import LearningRateScheduler
from keras import backend as K

import sys
import os
import argparse

import numpy as np

from sklearn.neural_network import MLPClassifier

from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.models import load_model
from keras import callbacks

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 20})
sess = tf.Session(config=config)

keras.backend.set_session(sess)
seed = 12306
np.random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def myargs():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data',
		required=False,
		default='../dataset/final_michailidu',
		type=int,
		help='the number of different p-value we want to try')
	parser.add_argument('--alpha_base',
		required=False,
		default=5e-3,
		type=float,
		help='the biggest p-value')

	parser.add_argument('--alpha_scale',
		required=False,
		default=6,
		type=int,
		help='the number of different p-value we want to try')
	parser.add_argument('--bs',
		required=False,
		default=64,
		help='batch size')
	parser.add_argument('--lr',
		required=False,
		default=0.00001,
		help='initial learning rate')
	parser.add_argument('--epoch',
		required=False,
		default=64,
		help='number of epochs for training')
	parser.add_argument('--output',
		required=False,
		default=None,
		help='location for the model to be saved')
	args = parser.parse_args()
	return args


epochs = 50
learning_rate = 0.01
decay_rate = 2e-6
momentum = 0.9
reg = 0.001

sd = []


class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = [1, 1]

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		sd.append(step_decay(len(self.losses)))
		print('learning rate:', step_decay(len(self.losses)))
		print('derivative of loss:', 2 * np.sqrt((self.losses[-1])))


def my_init(shape, name=None):
	value = np.random.random(shape)
	return K.variable(value, name=name)


def step_decay(history,losses):
	if float(2 * np.sqrt(np.array(history.losses[-1]))) < 1.1:
		lrate = 0.01 * 1 / (1 + 0.1 * len(history.losses))
		momentum = 0.2
		decay_rate = 0.0
		return lrate
	else:
		lrate = 0.01
		return lrate


class MLP_model():
	model = Sequential()
	model.add(Dense(16, input_dim=17, init=my_init))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(2, init='uniform'))
	model.add(Activation('softmax'))


def train():
	model = MLP_model()

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.97, nesterov=True)

	model.compile(loss='categorical_crossentropy',	optimizer=sgd, metrics=['accuracy'])

	aa = pd.read_csv('questao1NN.csv', sep=',', header=0)

	y_train = np.array(pd.get_dummies(aa['Desligamento'][0:1800]))
	y_test = np.array(pd.get_dummies(aa['Desligamento'][1801:1999]))

	X_train = np.array(aa.ix[:, 2:14][0:1800])
	X_test = np.array(aa.ix[:, 2:14][1801:1999])

	history = LossHistory()
	lrate = LearningRateScheduler(step_decay(history))

	model.fit(X_train,
			y_train,
			nb_epoch=epochs,
			batch_size=100,
			callbacks=[history, lrate])

	model.evaluate(X_train, y_train, batch_size=16)
	model.predict(X_train, batch_size=32, verbose=0)
	pred2 = model.predict_classes(X_test, batch_size=32, verbose=0)

	print('Accuracy Test Set=', len(np.where(pred2 - aa['Desligamento'][1801:1999]==0)[0]) / 198)

	''' Must apply a threshold to deal wih imbalanced classes'''


def load_data(folder):

	adenine_len = len(adenines)
	other_len = len(others)

	L = adenine_len * a_times + other_len * o_times

	voxel = np.zeros(shape=(L, 14, 32, 32, 32),
		dtype=np.float64)
	label = np.zeros(shape=(L,), dtype=int)
	cnt = 0
	numm = 0

	folder_list = os.listdir(voxel_folder)
	folder_list = [f for f in folder_list if 'rotate_voxel_data' in f]
	folder_list.sort()
	a_f_list = folder_list[0:a_times]
	o_f_list = folder_list[0:o_times]
	print('...Loading the data')
	print(len(a_f_list))
	print(len(o_f_list))

	for folder in a_f_list:
		for filename in os.listdir(voxel_folder + folder):
			ll = filename[0:-4].split('_')
			protein_name = ll[0] + "_" + ll[1]
			full_path = voxel_folder + folder + '/' + filename

			if protein_name in adenines:
				temp = np.load(full_path)
				voxel[cnt, :] = temp
				label[cnt] = 1
				cnt = cnt + 1
				numm = numm + 1
				print(numm, end=' ')
				if numm % 20==0:
					print()

	print('the adenine list done')
	num = 0
	for folder in o_f_list:
		for filename in os.listdir(voxel_folder + folder):
			ll = filename[0:-4].split('_')
			protein_name = ll[0] + "_" + ll[1]
			full_path = voxel_folder + folder + '/' + filename

			if protein_name in others:
				temp = np.load(full_path)
				voxel[cnt, :] = temp
				label[cnt] = 0
				cnt = cnt + 1
				num = num + 1
				print(num, end=' ')
				if num % 20==0:
					print()

	print('the other list done')
	print("total " + str(numm + num) + ' ligands')
	return voxel, label


def load_valid_data(adenines, others, voxel_folder, a_times, o_times):
	adenine_len = len(adenines)
	other_len = len(others)

	L = adenine_len * a_times + other_len * o_times

	voxel = np.zeros(shape=(L, 14, 32, 32, 32),
		dtype=np.float64)
	label = np.zeros(shape=(L,), dtype=int)
	cnt = 0
	numm = 0
	print('...Loading valid data')

	for filename in os.listdir(voxel_folder):
		if a_times==0:
			break
		ll = filename[0:-4].split('_')
		protein_name = ll[0] + "_" + ll[1]
		full_path = voxel_folder + '/' + filename

		if protein_name in adenines:
			temp = np.load(full_path)
			voxel[cnt, :] = temp
			label[cnt] = 1
			cnt = cnt + 1
			numm = numm + 1
			print(numm, end=' ')
	print('the adenine list done')
	num = 0

	for filename in os.listdir(voxel_folder):
		if o_times==0:
			break
		ll = filename[0:-4].split('_')
		protein_name = ll[0] + "_" + ll[1]
		full_path = voxel_folder + '/' + filename

		if protein_name in others:
			temp = np.load(full_path)
			voxel[cnt, :] = temp
			label[cnt] = 0
			cnt = cnt + 1
			num = num + 1
			print(num, end=' ')

	print('the other list done')

	print("valid data total " + str(numm + num) + ' ligands')
	return voxel, label


def train_deepdrug(folder, batch_size, lr, epoch, output):
	voxel_output = './data_prepare/valid/rotate_voxel_data_y_090/'

	mdl = MLPClassifier(solver='lbfgs',
		alpha=1e-5,
		hidden_layer_sizes=(5, 2),
		random_state=1)
	# mdl=keras.models.load_model('./deepdrug3d.h5')
	# mdl = multi_gpu_model(mdl,gpus=2)
	print(mdl.summary())

	adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

	# We add metrics to get more results you want to see
	mdl.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	# load the data
	adenines = []
	with open(adenine_list) as adenine_in:
		for line in adenine_in.readlines():
			temp = line.replace(' ', '').replace('\n', '')
			adenines.append(temp)
	others = []
	with open(other_list) as other_in:
		for line in other_in.readlines():
			temp = line.replace(' ', '').replace('\n', '')
			others.append(temp)
	# convert data into a single matrix
	a_times = 1  # 3  # the times rotate data needed, rotate 0, rotate 90, rotate 180
	o_times = 0  # 2  # the times rotate data needed, rotate 0, rotate 90
	voxel, label = load_data(adenines, others, voxel_folder, a_times, o_times)
	y = np_utils.to_categorical(label, num_classes=2)

	valid_voxel, valid_label = load_valid_data(adenines, others, voxel_output, 1, 0)
	v_y = np_utils.to_categorical(valid_label, num_classes=2)

	# print(voxel.shape)
	# print(label.shape)
	# print(label)
	earlyStopping = EarlyStopping(monitor='val_loss',
		patience=80,
		verbose=0,
		mode='min')
	mcp_save = ModelCheckpoint('.mdl_wts.hdf5',
		save_best_only=True,
		monitor='val_loss',
		mode='min')
	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
		factor=0.2,
		patience=15,
		verbose=1,
		min_delta=1e-4,
		mode='min')

	epo = 1
	times = epoch / epo
	print('times:' + str(times))
	for i in range(times):
		print('stage: ' + str(i))
		id = int(i)
		X_train, X_test, y_train, y_test = train_test_split(voxel, y, test_size=0.2, random_state=id)
		mdl.fit(X_train,
			y_train,
			epochs=epo,
			batch_size=batch_size,
			shuffle=True,
			validation_data=(valid_voxel, v_y),
			callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
			verbose=2)

		scores = mdl.evaluate(X_test, y_test, verbose=1)
		print(scores)

	if output==None:
		mdl.save('deepdrug3d.h5')
	else:
		mdl.save(output)


if __name__=="__main__":
	args = myargs()
	# args = argdet()
	train_mlp(args.vfolder, args.bs, args.lr, args.epoch, args.output)
