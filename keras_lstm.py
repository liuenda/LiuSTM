# coding: utf-8

"""
created on 2017/06/02
@author: liuenda
"""

import numpy as np
import pandas as pd

import keras

from sklearn.metrics import classification_report
from gensim import corpora, models, similarities
from sklearn import datasets
from sklearn import linear_model
from sklearn import svm
from gensim.models import word2vec
from sklearn import preprocessing
import random
import pickle
import time
from keras.preprocessing import sequence
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

base_path = "/home/liuenda/Workspace/2017.1.1~LiuSTM/"
model_name_en = base_path + "data/model-en/W2Vmodle.bin"
model_name_jp = base_path + "data/model-jp/W2Vmodle.bin"

model_en = word2vec.Word2Vec.load(model_name_en)
model_jp = word2vec.Word2Vec.load(model_name_jp)


maxlen = 300 # Default: 0 -> infinite
epoch = 50
random.seed(1234)

" Padding the sequence"
def padding(sequence, maxlen=300, padding_value=0.0):
	np_sequance = np.array(sequence)
	# print(np_sequance.shape)
	if np_sequance.shape[0] < maxlen:
		z = np.zeros((maxlen, 200))
		z[:np_sequance.shape[0], :np_sequance.shape[1]] = np_sequance
	else:
		z = np_sequance[:maxlen, :]
	return z



"""
Find the ranking results with respect to real pairs
Defaulty, projection1 should be JP
Whiile, projection2 should be EN->JP
"""
def find_ranking(projection1, projection2, dlmodel):
	sim_results = []
	rank_results = []

	# Iterate each of the ariticle from projection1 (999) as proj1
	# Calculate the simialrity of proj1 with all ariticles in projection2 (999)
	for i, proj1 in enumerate(projection1):
		print("Find answer for doc.", i)
		proj1_tile = np.tile(proj1, (len(projection2), 1, 1))
		sim = dlmodel.predict([proj1_tile, projection2])[:,0]
		rank = pd.Series(sim).rank(ascending = False)[i]
		sim_results.append(sim)
		rank_results.append(rank)

	# sim_results contains 999*999 similairty matrix
	return sim_results, rank_results

"""
rank_results should be list of (999,)
"""
def find_top(rank_results, top):
	s = pd.Series(rank_results)
	n_top = (s <= top).sum()
	return n_top


def average_docment(document_embedding):
	return np.average(document_embedding, axis=0)

def sum_docment(document_embedding):
	return np.sum(document_embedding, axis=0)

def doc2feature(corpus, tfidf, dictionary, w2v):
    doc_features = []
    for index, doc_bof in enumerate(corpus):

        if index % 1000 == 0:
            print(index)

        doc_tfidf = tfidf[doc_bof]

        doc_feature = np.zeros((200,))

        for (token_id, token_tfidf) in doc_tfidf:
            token = dictionary.get(token_id, "[unknown-id]").encode("utf-8")
            # if token in w2v:
            if True:
                token_w2v = w2v[token]
            else:
                print("No word:", token)
                continue
            doc_feature += token_w2v * token_tfidf

        average = True
        if average:
            doc_feature = np.true_divide(doc_feature, len(doc_tfidf))
        doc_features.append(doc_feature)

    return doc_features

def doc2vec_en(doc):
	r = [model_en[token] for token in doc.split()]
	return r

def doc2vec_jp(doc):
	r = [model_jp[token] for token in doc.split()]
	return r


def prepare_train(dir_en, dir_jp):

	df_en_mapping = pd.read_csv(dir_en)
	df_jp_mapping = pd.read_csv(dir_jp)

	print("Reading english Data:", len(df_en_mapping))
	print("Reading english Data:", len(df_jp_mapping))

	sample_size = len(df_en_mapping)

	assert len(df_en_mapping) == len(df_jp_mapping)

	# Convert mapping to list type and then concat to the a list
	print("Merging the English and Japanes news dataframe...")
	df_train_1 = pd.concat([df_en_mapping, df_jp_mapping], axis = 1)
	df_train_1['similarity'] = pd.Series(np.ones(sample_size,)*5)
	df_train_1['dis_similarity'] = pd.Series(np.ones(sample_size,)*1)

	# Remove null line
	print("Drop the null line...")
	# df_train_1 = df_train_1.dropna(subset=['en_article'])
	df_train_1 = df_train_1[df_train_1['en_article'] != '<NULL>']

	# Expand the training data
	en_article_wrong = df_train_1.en_article.iloc[random.sample(range(len(df_train_1)),len(df_train_1))]
	en_article_wrong.index = df_train_1.index
	print((en_article_wrong == df_train_1.en_article).value_counts())
	df_train_1['en_article_wrong'] = en_article_wrong

	# Convert dateframe to list
	train_1 = df_train_1[['en_article','jp_article','similarity']].values.tolist()
	train_2 = df_train_1[['en_article_wrong','jp_article','dis_similarity']].values.tolist()

	return train_1, train_2, df_train_1

if __name__ == "__main__":


	input = 2
	k = 10

	# --- Prepare and Loading the training data --- #

	if input == 1:
		# Prepare For the training data
		sample_size = "_1000"
		dir_en = "./data/mapping/en_mapped_"+str(k) + sample_size + ".csv"
		dir_jp = "./data/mapping/jp_mapped_" + str(k) + sample_size + ".csv"

		# Prepare For the test data
		sample_size = "_1k2k"
		dir_en_test = "./data/mapping/en_mapped_"+str(k) + sample_size + ".csv"
		dir_jp_test = "./data/mapping/jp_mapped_" + str(k) + sample_size + ".csv"

		train_1, train_2, df_train_1 = 	prepare_train(dir_en, dir_jp)
		test_1, test_2, df_test_1 = prepare_train(dir_en_test, dir_jp_test)

	if input == 2:
		# split_line = 5000
		# end_line = 6000
		# Prepare For the training data
		dir_en = base_path + "en_news.csv"
		dir_jp = base_path + "jp_news.csv"

		pairs_correct, pairs_wrong, df_pairs = prepare_train(dir_en, dir_jp)
		train_1 = pairs_correct[0:2000] + pairs_correct[3000:5000]
		test_1 = pairs_correct[2000:3000]

		train_2 = pairs_wrong[0:2000] + pairs_wrong[3000:5000]
	# test_2 = pairs_wrong[split_line:end_line]


	# Expand the training data
	train = train_1 + train_2


	# --- Apply the word2vec model to the data sets --- #

	df_pairs_sample = df_pairs.iloc[0:5000]

	df_pairs_sample['word2vec_en'] = df_pairs_sample['en_article'].apply(doc2vec_en)
	df_pairs_sample['word2vec_jp'] = df_pairs_sample['jp_article'].apply(doc2vec_jp)


	# ---- Padding the vector ---- #
	df_pairs_sample['padding_en'] = df_pairs_sample['word2vec_en'].apply(padding)
	df_pairs_sample['padding_jp'] = df_pairs_sample['word2vec_jp'].apply(padding)


	# --- Prepare the training data --- #

	# Generate training data (similarity = 1)
	features_en_1 = np.stack(df_pairs_sample["padding_en"].values)
	features_jp_1 = np.stack(df_pairs_sample["padding_jp"].values)

	# Generate training data (similarity = 0)
	features_en_0 = np.array(features_en_1)
	np.random.shuffle((features_en_0))

	# check the duplicated amount
	c = np.all(features_en_1 == features_en_0, axis=(1,2))
	print "C value =", c.sum(), "position:", np.where(c== True)[0].tolist()

	# Prepare the final training and test data
	X_1 = np.concatenate((features_en_1, features_en_0), axis = 0)
	X_2 = np.concatenate((features_jp_1, features_jp_1), axis = 0)
	y = np.concatenate((np.ones(len(features_en_1)), np.zeros(len(features_en_0))), axis = 0)

	# --- Split into test data and training data --- #

	X1_train1, X1_test_1, X1_train2, X1_train3_wrong, X1_test_0 = np.split(X_1, [2000, 3000, 5000, 9000])
	X2_train1, X2_test_1, X2_train2, X2_train3_wrong, X2_test_0 = np.split(X_2, [2000, 3000, 5000, 9000])
	y_train1, y_test, y_train2, y_train3_wrong, Y_o = np.split(y, [2000, 3000, 9000, 9000])

	X1_train = np.concatenate((X1_train1, X1_train2, X1_train3_wrong), axis = 0)
	X2_train = np.concatenate((X2_train1, X2_train2, X2_train3_wrong), axis = 0)
	y_train = np.concatenate((y_train1, y_train2, y_train3_wrong), axis = 0)
	# X_train_correct = np.concatenate((X_train1, X_train2), axis = 0)
	# y_train_correct = np.concatenate((y_train1, y_train2), axis = 0)

	# --- Generate balanced test data --- #
	X1_test = np.concatenate((X1_test_1, X1_test_0), axis=0)
	X2_test = np.concatenate((X2_test_1, X2_test_0), axis=0)
	y_test = np.concatenate((np.ones(len(X1_test_1)), np.zeros(len(X1_test_0))), axis = 0)


	# ---- Parallel Model ---- #

	# Input layer
	input_1 = Input(shape=(maxlen,200), dtype='float32', name='main_input_1')
	input_2 = Input(shape=(maxlen,200), dtype='float32', name='main_input_2')

	# LSTM layer
	# lstm_out_1 = LSTM(50)(input_1)
	# lstm_out_2 = LSTM(50)(input_2)
	lstm_out_1 = LSTM(50, go_backwards = True)(input_1)
	lstm_out_2 = LSTM(50, go_backwards = True)(input_2)

	# Merge layer
	merged_vector = keras.layers.concatenate([lstm_out_1, lstm_out_2], axis=-1)

	# (Dense 1) * 3
	x1 = Dense(64, activation='relu')(merged_vector)
	x1 = Dense(64, activation='relu')(x1)
	x1 = Dense(64, activation='relu')(x1)
	main_output = Dense(1, activation='sigmoid', name='main_output')(x1)

	# Model definition
	model_lstm2 = Model(input=[input_1, input_2], output=main_output)

	# Compile the model
	model_lstm2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


	# Fit the training model
	hist = model_lstm2.fit([X1_train, X2_train], [y_train],
	                       validation_data=([X1_test, X2_test], y_test), epochs=50, batch_size=256)

	# Save the history and the model
	code = "b"
	path_model_lstm2 = "model_lstm2_" + code
	model_lstm2.save(path_model_lstm2)
	path_hist = "hist_lstm2_" + code
	f = open(path_hist, "wb")
	pickle.dump(hist.history, f)
	f.close()


	# --- find ranking --- #
	find_ranking(X1_test_1, X2_test_1, model_lstm2)


	# # --- Prepare for a new independent evaluation balanced data --- #
	#
	# df_pairs_evaluate = df_pairs.iloc[55000:60000]
	#
	# df_pairs_evaluate['word2vec_en'] = df_pairs_evaluate['en_article'].apply(doc2vec_en)
	# df_pairs_evaluate['word2vec_jp'] = df_pairs_evaluate['jp_article'].apply(doc2vec_jp)
	#
	# features_en_eva = doc2feature(corpus_en[60000:61000], tfidf_en, dictionary_en, model_en)
	# features_jp_eva = doc2feature(corpus_jp[60000:61000], tfidf_jp, dictionary_jp, model_jp)
	#
	# features_merge_eva = np.concatenate((features_en_eva,features_jp_eva), axis = 1)
	#
	# features_en_wrong_eva =  features_en[:1000]
	# # features_en_wrong_eva = np.array(features_en_eva)
	# # np.random.shuffle((features_en_wrong_eva))
	# # c = np.all(features_en_wrong_eva == features_en_eva, axis=1)
	# # print "C value =", c.sum() # check the duplicated amount
	#
	# features_merge_wrong = np.concatenate((features_en_wrong_eva,features_jp_eva), axis = 1)
	#
	# X_eva = np.concatenate((features_merge_eva, features_merge_wrong), axis = 0)
	# y_eva = np.concatenate((np.ones(len(features_merge_eva)), np.zeros(len(features_en_wrong_eva))), axis = 0)
	#
	# y_eva_predict = clf.predict(X_eva)
	#
	# print("classification report of TRAINING data:")
	# print(classification_report(y_eva, y_eva_predict))
	#
	#
	# # --- Evaluation for SVM --- #
	#
	# y_test_proba = clf.predict_proba(X_test)
	# y_train_proba = clf.predict_proba(X_train)
	#
	# # sim_results_train, rank_results_train = find_ranking(projection1_train, projection2_train)
	# sim_results_test, rank_results_test = find_ranking(X_test[:,:200] ,X_test[:,200:], clf)
	#
	#
	# print(pd.Series(rank_results_test).describe())



