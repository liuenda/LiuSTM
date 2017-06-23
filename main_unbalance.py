# coding: utf-8
import random
import pickle
import time
import pandas as pd
import numpy as np
import lstm_proj as lstm

maxlen = 0  # Default: 0 -> infinite
epoch = 50
random.seed(1234)


def prepare_train(dir_en, dir_jp):
	df_en_mapping = pd.read_csv(dir_en)
	df_jp_mapping = pd.read_csv(dir_jp)

	print "Reading english Data:", len(df_en_mapping)
	print "Reading english Data:", len(df_jp_mapping)

	sample_size = len(df_en_mapping)

	assert len(df_en_mapping) == len(df_jp_mapping)

	# Convert mapping to list type and then concat to the a list
	print "Merging the English and Japanes news dataframe..."
	df_train_1 = pd.concat([df_en_mapping, df_jp_mapping], axis=1)
	df_train_1['similarity'] = pd.Series(np.ones(sample_size, ) * 5)
	df_train_1['dis_similarity'] = pd.Series(np.ones(sample_size, ) * 1)

	# Remove null line
	print "Drop the null line..."
	# df_train_1 = df_train_1.dropna(subset=['en_article'])
	df_train_1 = df_train_1[df_train_1['en_article'] != '<NULL>']

	# Expand the training data
	en_article_wrong = df_train_1.en_article.iloc[random.sample(xrange(len(df_train_1)), len(df_train_1))]
	en_article_wrong.index = df_train_1.index
	print (en_article_wrong == df_train_1.en_article).value_counts()
	df_train_1['en_article_wrong'] = en_article_wrong

	# Convert dateframe to list
	train_1 = df_train_1[['en_article', 'jp_article', 'similarity']].values.tolist()
	train_2 = df_train_1[['en_article_wrong', 'jp_article', 'dis_similarity']].values.tolist()

	return train_1, train_2, df_train_1

"""
def word_embedding(a_sentence, model):
	embedding = [get_vector(word, model) for word in a_sentence.split()]
	return embedding


def get_vector(word, model):
	word = word.rstrip()  # remove all '\n' and '\r'
	# word=word.lower()
	# baseform=getVector.getBase(word,wnl)
	# print "DEBUG: ",model['good']
	# print "DEBUG: baseform= ", baseform
	try:
		vecW = model[word]  # !!!Maybe the word is not existed
	except Exception, e:
		# info=''
		# counter_NaN+=1 #increase 1 to NaN counter
		# info+=repr(e)+"\n" #create log information
		# logout.write(info) #write log information to log file
		# new 3.15: generate a useless list for deleting in the next stage
		output_unmatch.write(word)  # no \n is needed since the
		output_unmatch.write('\n')
		print "---Warning: Word [" + word + "] Vector Not Found ---"
		return nan
	else:
		# vecW=getVector.vecNorm(vecW) #Normalized the raw vector
		# print "(the new length of the vector is:",LA.norm(vecW),")"
		# info+=baseform+": OK!\n" #create log information
		# logout.write(info) #write log information to log file
		# fout.write(rawVoc) #add in 16/3/17
		# good_list.append(rawVoc)
		# append the new vector to the matrix
		# if the vector is the first element in the matrix: 'good_vecs', reshape it
		return vecW


"""
def read_vecs(lang_name):
	filename = './data_baseline/good_vecs_' + lang_name + '.csv'
	print "[INFO]Reading the word2vec vectors in ", lang_name, " from ", filename, "---"
	df = pd.read_csv(filename)
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
Projection matrix A, for J -> E
"""
def fit_projection(J, E):
	JTJ = np.dot(J.T, J)

	# iJTJ=np.linalg.inv(np.matrix(JTJ))
	# print np.dot(JTJ,iJTJ) # This is now identical!!
	# print np.linalg.det(JTJ)  # The determinant of det(JTJ) is 0!
	# Which means there is no inverse matrix

	# Solutions: Using linear ridge regression
	la = 0.00001
	dim = 200
	I = np.eye(dim)
	print "[DEBUG]", "inv(JTJ)*JTJ=I?: "
	print "[DEBUG]", (JTJ + la * I).dot(np.linalg.inv(JTJ + la * I))
	W = np.dot(np.linalg.inv(JTJ + la * I), J.T).dot(E)

	return W


if __name__ == "__main__":

	df_E = read_vecs('en')
	df_J = read_vecs('jp')

	E = df_E.drop('en', axis=1)
	J = df_J.drop('jp', axis=1)

	# Note: W is the projection from E to J
	W = fit_projection(E, J)

	# root_dir = "./pickles/"
	# df_train_1 = pickle.load(open(root_dir + "df_train_1.p",'rb'))
	# df_test_1 = pickle.load(open(root_dir + "df_test_1.p",'rb'))

	k = 10
	input = 1

	if input == 1:
		# Prepare For the training data
		sample_size = "_1000"
		dir_en = "./data/mapping/en_mapped_" + str(k) + sample_size + ".csv"
		dir_jp = "./data/mapping/jp_mapped_" + str(k) + sample_size + ".csv"

		# Prepare For the test data
		sample_size = "_1k2k"
		dir_en_test = "./data/mapping/en_mapped_" + str(k) + sample_size + ".csv"
		dir_jp_test = "./data/mapping/jp_mapped_" + str(k) + sample_size + ".csv"

		train_1, train_2, df_train_1 = prepare_train(dir_en, dir_jp)
		test_1, test_2, df_test_1 = prepare_train(dir_en_test, dir_jp_test)

	if input == 2:
		# split_line = 5000
		# end_line = 6000
		# Prepare For the training data
		dir_en = "./data/news/en_news.csv"
		dir_jp = "./data/news/jp_news.csv"

		pairs_correct, pairs_wrong, df_pairs = prepare_train(dir_en, dir_jp)
		train_1 = pairs_correct[0:2000] + pairs_correct[3000:5000]
		test_1 = pairs_correct[2000:3000]

		train_2 = pairs_wrong[0:2000] + pairs_wrong[3000:5000]
	# test_2 = pairs_wrong[split_line:end_line]


	# Expand the training data
	train = train_1 + train_2

	# True to training the data, False to laod the existed data
	print "Now the maxlen =", maxlen
	batchsize = 256
	if True:
		dir_file = "weights/proj/20170326_e50_4000_b256"
		print "Starting to training the model..., saving to", dir_file
		sls = lstm.LSTM(dir_file, W, maxlen=maxlen, load=False, training=True)
		sls.train_lstm(train, epoch, train_1, test_1, batchsize=batchsize)
		sls.save_model()
	else:
		dir_file = "weights/proj/20170320_e50_4000_b64.p"
		print "NO Training. Load the existed model:", dir_file
		sls = lstm.LSTM(dir_file, W, maxlen=maxlen, load=True, training=False)

	if True:
		print "Evaluate the model using fast estimation..."
		projection1_train, projection2_train = sls.seq2vec(train_1)
		projection1_test, projection2_test = sls.seq2vec(test_1)

		sim_results_train, rank_results_train = lstm.find_ranking(projection1_train, projection2_train)
		sim_results_test, rank_results_test = lstm.find_ranking(projection1_test, projection2_test)

		print pd.Series(rank_results_train).describe()
		print pd.Series(rank_results_test).describe()
