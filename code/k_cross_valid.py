from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import math
import keras

from keras.optimizers import Adam, SGD
import os
import argparse
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


seed = 7
np.random.seed(seed)


def myargs():
	parser = argparse.ArgumentParser()
	parser.add_argument('--folder',
		required=False,
		default='../output/',
		help='the input data set folder')
	parser.add_argument('--alpha_base',
		required=False,
		default=5,
		type=float,
		help='the biggest p-value')

	parser.add_argument('--alpha_scale',
		required=False,
		default=3,
		type=int,
		help='the number of different p-value we want to try')
	parser.add_argument('--bs',
		required=False,
		default=32,
		help='batch size')
	parser.add_argument('--lr',
		required=False,
		default=0.00001,
		help='initial learning rate')
	parser.add_argument('--epoch',
		required=False,
		default=90,
		help='number of epochs for training')
	parser.add_argument('--output',
		required=False,
		default=None,
		help='location for the model to be saved')
	args = parser.parse_args()
	return args


def create_model():
	model = Sequential()
	model.add(Dense(32, input_dim=17, kernel_initializer='random_uniform'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	# model.add(Activation('tanh'))
	# Dropout 1
	model.add(Dropout(0.2))

	model.add(Dense(64, kernel_initializer='uniform'))
	model.add(BatchNormalization())
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.3))

	model.add(Dense(64, kernel_initializer='random_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Dropout(0.4))

	model.add(Dense(16, kernel_initializer='uniform'))
	model.add(BatchNormalization())
	# model.add(LeakyReLU(alpha=0.1))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))

	model.add(Dense(2, kernel_initializer="random_uniform"))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))
	adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

	# We add metrics to get more results you want to see
	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
	return model


def read_file(file):
	myfile = open(file, 'r')
	content = myfile.readlines()
	content = [x.strip().split('\t')[0:-1] for x in content]
	n_len = len(content)
	return content, n_len


def load_data(folder, p_val, val_rate):
	n_len = 0
	n_content = []
	n_label = []
	p_content = []
	p_len = 0
	p_label = []
	for fld in os.listdir(folder):
		if fld=="negative":
			for filenames in os.listdir(os.path.join(folder, fld)):
				tmp_path = os.path.join(folder, fld)
				tmp_file = tmp_path + '/' + filenames
				n_content, n_len = read_file(tmp_file)
				n_label = np.zeros(shape=(n_len,), dtype=int)

		if fld=="positive":
			for filenames in os.listdir(os.path.join(folder, fld)):
				pvalue = float(filenames.split('_')[1])
				if abs(pvalue - p_val) < 1e-10:
					tmp_path = os.path.join(folder, fld)
					tmp_file = tmp_path + '/' + filenames
					p_content, p_len = read_file(tmp_file)
					p_label = np.ones(shape=(p_len,), dtype=int)

	L = p_len + n_len
	print('total length: ', L)
	val_num = int(val_rate * L)
	val_num_n = int(val_num / 2)
	val_num_p = val_num - val_num_n
	data = np.vstack((np.array(n_content[0:-val_num_n]), np.array(p_content[0:-val_num_p])))
	label = np.hstack((n_label[0:-val_num_n], p_label[0:-val_num_p]))

	val_data = np.vstack((np.array(n_content[-val_num_n:]), np.array(p_content[-val_num_p:])))
	val_label = np.hstack((n_label[-val_num_n:], p_label[-val_num_p:]))
	return data, label, val_data, val_label


def train_mlp(folder, p_val, batch_size, lr, epoch, output, weight_n, weight_p, val_rate):
	# mdl = MLPClassifier(solver='lbfgs',
	# 	alpha=1e-5,
	# 	hidden_layer_sizes=(5, 2),
	# 	random_state=1)
	# mdl=keras.models.load_model('./deepdrug3d.h5')
	# mdl = MLP_model.build()

	# # mdl = multi_gpu_model(mdl, gpus=2)
	# print(mdl.summary())
	#
	# adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
	#
	# # We add metrics to get more results you want to see
	# mdl.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	model = create_model()
	# model = KerasClassifier(build_fn=create_model, verbose=0)
	vector, label, valid_data, valid_label = load_data(folder, p_val, val_rate)
	y = np_utils.to_categorical(label, num_classes=2)

	v_y = np_utils.to_categorical(valid_label, num_classes=2)

	feature_scaler = StandardScaler()
	vector = feature_scaler.fit_transform(vector)
	valid_data = feature_scaler.transform(valid_data)
	print(vector.shape)
	print(label.shape)
	print(label)

	earlyStopping = EarlyStopping(monitor='val_loss',
		patience=10,
		verbose=0,
		mode='min')
	mcp_save = ModelCheckpoint('.mdl_wts.hdf5',
		save_best_only=True,
		monitor='val_loss',
		mode='min')
	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
		factor=0.5,
		patience=5,
		verbose=1,
		min_delta=1e-4,
		mode='min')

	epo = 10
	times = int(epoch / epo)
	print('times:' + str(times))
	class_wgt = {0: weight_n,
	             1: weight_p}
	for i in range(times):
		print('stage: ' + str(i))
		id = int(i)
		X_train, X_test, y_train, y_test = train_test_split(vector, y, test_size=0.2, random_state=id, shuffle=True)

		model.fit(X_train,
			y_train,
			epochs=epo,
			batch_size=batch_size,
			shuffle=True,
			class_weight=class_wgt,
			validation_data=(valid_data, v_y),
			callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
			verbose=2)

		scores = model.evaluate(X_test, y_test, verbose=1)
		print(scores)
	res = model.fit(vector, y)
	print(res)
	if output==None:
		model.save('MLP_GWAS.h5')
	else:
		model.save(output)


def train_mlp_hyper_p(folder, p_val, batch_size, lr, epoch, output, weight_n, weight_p, val_rate):
	# create model
	# model = MLP_model.build()
	# adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
	# model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	model = KerasClassifier(build_fn=create_model, verbose=0)
	vector, label, valid_data, valid_label = load_data(folder, p_val, val_rate)
	y = np_utils.to_categorical(label, num_classes=2)
	v_y = np_utils.to_categorical(valid_label, num_classes=2)
	feature_scaler = StandardScaler()
	vector = feature_scaler.fit_transform(vector)
	valid_data = feature_scaler.transform(valid_data)
	# scores = cross_val_score(model, vector, y, cv=5)

	# define the grid search parameters
	tuned_parameters = {'batch_size': [32],
	                    'epochs': [90],
	                    'callbacks': [[EarlyStopping(monitor='val_loss', patience=7, mode='min'),
	                                  ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss',
		                                  mode='min'),
	                                  ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1,
		                                  in_delta=1e-4, mode='min')
	                                  ]],
	                    'verbose': [2],

	                    'validation_data': [(valid_data, v_y)],
	                    'class_weight':[{0: weight_n, 1: weight_p}]
	                    }

	grid = GridSearchCV(model, tuned_parameters, cv=10, shuffle=True)
	grid_result = grid.fit(vector, y)
	print(grid)


	# # summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))


if __name__=="__main__":
	# load dataset

	args = myargs()

	p_val = args.alpha_base * math.pow(10, -args.alpha_scale)
	val_rate = 0.005
	weight_0 = 1
	weight_1 = 20.
	# train_mlp_hyper_p(args.folder, p_val, args.bs, args.lr, args.epoch, args.output, weight_0, weight_1, val_rate)
	train_mlp(args.folder, p_val, args.bs, args.lr, args.epoch, args.output, weight_0, weight_1, val_rate)