# coding: utf-8
import time
import pandas as pd
import numpy as np


def	word_embedding(a_sentence, model):
	embedding = [get_vector(word, model) for word in a_sentence.split()]
	return embedding

def get_vector(word,model):
	word=word.rstrip() # remove all '\n' and '\r'
	# word=word.lower()
	# baseform=getVector.getBase(word,wnl)
	# print "DEBUG: ",model['good']
	# print "DEBUG: baseform= ", baseform
	try:
		vecW=model[word] #!!!Maybe the word is not existed
	except Exception,e:
		# info=''
		# counter_NaN+=1 #increase 1 to NaN counter
		# info+=repr(e)+"\n" #create log information
		# logout.write(info) #write log information to log file				
		#new 3.15: generate a useless list for deleting in the next stage
		output_unmatch.write(word) # no \n is needed since the 
		output_unmatch.write('\n')
		print "---Warning: Word ["+word+"] Vector Not Found ---"
		return nan
	else:
		# vecW=getVector.vecNorm(vecW) #Normalized the raw vector
		# print "(the new length of the vector is:",LA.norm(vecW),")"
		# info+=baseform+": OK!\n" #create log information
		# logout.write(info) #write log information to log file
		# fout.write(rawVoc) #add in 16/3/17
		# good_list.append(rawVoc)
		#append the new vector to the matrix
		#if the vector is the first element in the matrix: 'good_vecs', reshape it
		return vecW

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

	# root_dir = "../pickles/"
	# df_train_1 = pickle.load(open(root_dir + "df_train_1.p",'rb'))
	# df_test_1 = pickle.load(open(root_dir + "df_test_1.p",'rb'))	

	# model_name_en = "../data/model-en/W2Vmodle.bin"
	# model_name_jp = "../data/model-jp/W2Vmodle.bin"

	# model_en = Word2Vec.load(model_name_en)
	# model_jp = Word2Vec.load(model_name_jp)

