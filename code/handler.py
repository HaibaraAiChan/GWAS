

import argparse
import math
from main_cross_valid import train_mlp
from predict_self import predict

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
		default=64,
		help='batch size')
	parser.add_argument('--lr',
		required=False,
		default=0.00001,
		help='initial learning rate')
	parser.add_argument('--epoch',
		required=False,
		default=100,
		help='number of epochs for training')
	parser.add_argument('--output',
		required=False,
		default='./result_model/model_',
		help='location for the model to be saved')
	args = parser.parse_args()
	return args


if __name__=="__main__":
	args = myargs()
	# args = argdet()
	p_val = args.alpha_base * math.pow(10,-args.alpha_scale)

	for i in range(10):
		train_mlp(args.folder,p_val, args.bs, args.lr, args.epoch, args.output+str(i), 0.082+0.002*i, 1.)
		predict(args.folder, p_val, args.output+str(i))