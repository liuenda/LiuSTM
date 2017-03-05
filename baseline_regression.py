# coding: utf-8
import pandas as pd
import numpy as np
import time
from ast import literal_eval
import random
import pickle
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def freq_count(appearance):
	a = pd.Series([int(n) for n in appearance.split()])
	d = a.value_counts().to_dict()
	r = [d[x] if x in d.keys() else 0 for x in range(1,11)]
	return r

	
if __name__ == "__main__":

	root_dir = "pickles/"
	# ------load the exited prepared data from pickle---------------
	train_1 = pickle.load(open(root_dir + "train_1.p",'rb'))
	train_2 = pickle.load(open(root_dir + "train_2.p",'rb'))
	test_1 = pickle.load(open(root_dir + "test_1.p",'rb'))
	# test_2 = pickle.load(open(root_dir + "test_2.p",'rb')) 
	df_train_1 = pickle.load(open(root_dir + "df_train_1.p",'rb'))
	df_test_1 = pickle.load(open(root_dir + "df_test_1.p",'rb'))

	df_train_1['xa_c'] = df_train_1['xa'].apply(freq_count)
	df_train_1['xb_c'] = df_train_1['xb'].apply(freq_count)
	df_train_1['xb_c_wrong'] = df_train_1['xb_wrong'].apply(freq_count)

	df_train_1['feature'] = df_train_1['xa_c'] + df_train_1['xb_c']
	df_train_1['feature_wrong'] = df_train_1['xa_c'] + df_train_1['xb_c_wrong']

	train_right = df_train_1[['feature','similarity']].values.tolist()
	train_wrong = df_train_1[['feature_wrong','dis_similarity']].values.tolist()

	X_train = np.array([i[0] for i in (train_right + train_wrong)])
	y_train = (np.array([i[1] for i in (train_right + train_wrong)])-1)/4

	df_test_1['xa_c'] = df_test_1['xa'].apply(freq_count)
	df_test_1['xb_c'] = df_test_1['xb'].apply(freq_count)
	df_test_1['xb_c_wrong'] = df_test_1['xb_wrong'].apply(freq_count)

	df_test_1['feature'] = df_test_1['xa_c'] + df_test_1['xb_c']
	df_test_1['feature_wrong'] = df_test_1['xa_c'] + df_test_1['xb_c_wrong']

	test_right = df_test_1[['feature','similarity']].values.tolist()
	test_wrong = df_test_1[['feature_wrong','dis_similarity']].values.tolist()

	X_test = np.array([i[0] for i in (test_right + test_wrong)])
	y_test = (np.array([i[1] for i in (test_right + test_wrong)])-1)/4

	scaler = preprocessing.StandardScaler().fit(X_train)

	# X_train = StandardScaler().fit_transform(X_train)	
	# X_test = StandardScaler().fit_transform(X_test)	
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test) 

	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(X_train, y_train)
	logreg.score(X_train, y_train)
	logreg.score(X_test, y_test)
