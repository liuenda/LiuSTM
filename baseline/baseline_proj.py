# coding: utf-8
import time
import pandas as pd
import numpy as np


def read_vecs(lang_name):
	filename='./data_baseline/good_vecs_'+lang_name+'.csv'
	print "[INFO]Reading the word2vec vectors in ",lang_name," from ",filename,"---"
	df=pd.read_csv(filename)
	return df


"""
Description:
Let J*A=E, where A is the projection matrix
Since A=inv(J.T*J)*J.T*E

Here, in order to avoid null results of inverse matrix
We impletment A = inv(J.T*J + la*I)*J.T*E

Parameters:
J: Matrix J
E: Marix E

Return:
Projection matrix A
"""
def fit_projection(J,E):
	JTJ = np.dot(J.T,J)

	# iJTJ=np.linalg.inv(np.matrix(JTJ))
	# print np.dot(JTJ,iJTJ) # This is now identical!!
	# print np.linalg.det(JTJ)  # The determinant of det(JTJ) is 0!
	# Which means there is no inverse matrix

	# Solutions: Using linear ridge regression 
	la = 0.00001
	dim = 200
	I = np.eye(dim)
	print "[DEBUG]","inv(JTJ)*JTJ=I?: "
	print "[DEBUG]", (JTJ + la * I).dot(np.linalg.inv(JTJ + la * I))
	W = np.dot(np.linalg.inv(JTJ + la * I),J.T).dot(E)

	return W


if __name__ == "__main__":

	df_E = read_vecs('en')
	df_J = read_vecs('jp')

	E = df_E.drop('en',axis=1)
	J = df_J.drop('jp',axis=1)
	W = fit_projection(E, J)

