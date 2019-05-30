# try different alpha value, 5e-3,5e-4,5e-5,5e-6
# each is positive and negative classification
# < alpha: positive
# > alpha: negative
# different alpha generate different data sets
import pandas as pd
import numpy as np

import re

import time


def data_gen(input_csv, output, p_value):
	df = pd.read_csv(input_csv, delim_whitespace=True)
	#
	temp_zip = zip(df.wildtype_value, df.Blosum62_wt, df.mutant_value, df.Blosum62_mu, df.disorder,
		df.confidence, df.core, df.NIS, df.Interface, df.HBO, df.SBR, df.Aromatic, df.Hydrophobic,
		df.helix, df.coil, df.sheet, df.Entropy, df.P_value)

	tmp_zip = temp_zip
	size = len(list(tmp_zip))
	print(size)
	f_pos_out = open(output + str(p_value) + '_pos.txt', 'w')
	f_neg_out = open(output + str(p_value) + '_neg.txt', 'w')
	w_pos_str = ''
	w_neg_str = ''
	for i in range(size):

		tmp_str = str(df.wildtype_value[i]) + '\t' \
	            + str(df.Blosum62_wt[i]) + '\t' \
	            + str(df.mutant_value[i])+ '\t'  \
	            + str( df.Blosum62_mu[i])+ '\t'  \
	            + str(df.disorder[i]) + '\t' \
	            + str(df.confidence[i]) + '\t' \
	            + str(df.core[i]) + '\t' \
	            + str(df.NIS[i])+ '\t'  \
	            + str(df.Interface[i])+ '\t'  \
	            + str(df.HBO[i]) + '\t' \
	            + str(df.SBR[i])+ '\t'  \
		        + str(df.Aromatic[i]) + '\t' \
		        + str(df.Hydrophobic[i]) + '\t' \
				+ str(df.helix[i]) + '\t' \
				+ str(df.coil[i]) + '\t' \
				+ str(df.sheet[i]) + '\t' \
		        + str(df.Entropy[i]) + '\t'\
				+ str(df.P_value[i])+ '\n'

				#
				# + str(df.Entropy[i])+'\n'

		if df.P_value[i] <= p_value:
			w_pos_str = w_pos_str + tmp_str
		if df.P_value[i] > p_value:
			w_neg_str = w_neg_str + tmp_str
		print(i)

	f_pos_out.write(w_pos_str)
	f_pos_out.close()
	f_neg_out.write(w_neg_str)
	f_neg_out.close()

if __name__=="__main__":
	input_csv = '../dataset/final_michailidu'
	output = '../output/original/alpha_'
	p_value = 5e-2
	# p_value = 5e-4
	# p_value = 5e-7

	start = time.time()

	data_gen(input_csv, output, p_value)
	end = time.time()
	print('time elapsed :' + str(end - start))
