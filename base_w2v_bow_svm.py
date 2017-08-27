# coding: utf-8

"""
created on 2017/05/11
@author: liuenda
"""

import numpy as np
import pandas as pd

from sklearn import cross_validation
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
import sys

from copy import deepcopy
from multiprocessing import Process, Value, Array
import multiprocessing

model_name_en = "./data/model-en/W2Vmodle.bin"
model_name_jp = "./data/model-jp/W2Vmodle.bin"

model_en = word2vec.Word2Vec.load(model_name_en)
model_jp = word2vec.Word2Vec.load(model_name_jp)

average = False # 这里对于文章分类来说必须是False，否则就会严重影响精度
random.seed(1234)
step = 100
print("average=", average)

def wrapper_find_ranking_quick(X_test_scaled, clf):
	res = {}
	queue = multiprocessing.Queue()
	queue.put(res)

	jobs = []
	l_clf = []

	n_jobs = 10
	for i in range(n_jobs):

		# # 1. A lower way but more memory
		#     l_clf.append(deepcopy(clf))
		#     jobs.append(Process(target=find_ranking_quick, args=((X_test_scaled[i*step:i*step+step,:200],
		#                                                           X_test_scaled[:100,200:],
		#                                                           l_clf[i], i, queue),)))

		# 2. A faster way but more memory
		jobs.append(Process(target=find_ranking_quick, args=((deepcopy(X_test_scaled[i * step:i * step + step, :200]),
		                                                      deepcopy(X_test_scaled[:1000, 200:]),
		                                                      deepcopy(clf), i, queue),)))
	s = time.time()
	for j in jobs:
		j.start()

	for j in jobs:
		j.join()

	print(time.time() - s)

	return  queue


def find_ranking_quick(args):
	sim_results = []
	rank_results = []
	step = 100
	projection1, projection2, clf, n, queue = args[0], args[1], args[2], args[3], args[4]
	res = {}

	# Iterate each of the ariticle from projection1 (999) as proj1
	# Calculate the simialrity of proj1 with all ariticles in projection2 (999)
	for i, proj1 in enumerate(projection1):
		# print "Find answer for doc.", i
		proj1_tile = np.tile(proj1, (len(projection2), 1))
		features_test = np.concatenate((proj1_tile, projection2), axis=1)
		sim = clf.predict_proba(features_test)[:, 1]
		rank = pd.Series(sim).rank(ascending=False)[n * step + i]
		sim_results.append(sim)
		rank_results.append(rank)
		# print("Find answer for doc.", n * step + i, rank, end="||", flush=True)
		sys.stdout.write("Doc." + str(n * step + i) + ": " + str(rank) + "||")
		sys.stdout.flush()
		res[n * step + i] = rank

	res_all = queue.get()
	res_all.update(res)
	queue.put(res_all)
	return sim_results, rank_results

"""
Find the ranking results with respect to real pairs
Defaulty, projection1 should be JP
Whiile, projection2 should be EN->JP
"""
def find_ranking(projection1, projection2, clf):
	sim_results = []
	rank_results = []

	# Iterate each of the ariticle from projection1 (999) as proj1
	# Calculate the simialrity of proj1 with all ariticles in projection2 (999)
	for i, proj1 in enumerate(projection1):
		print "Find answer for doc.", i
		proj1_tile = np.tile(proj1, (len(projection2),1))
		features_test = np.concatenate((proj1_tile, projection2), axis=1)
		sim = clf.predict_proba(features_test)[:,1]
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

        # average = False
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

	print "Reading english Data:", len(df_en_mapping)
	print "Reading english Data:", len(df_jp_mapping)

	sample_size = len(df_en_mapping)

	assert len(df_en_mapping) == len(df_jp_mapping)

	# Convert mapping to list type and then concat to the a list
	print "Merging the English and Japanes news dataframe..."
	df_train_1 = pd.concat([df_en_mapping, df_jp_mapping], axis = 1)
	df_train_1['similarity'] = pd.Series(np.ones(sample_size,)*5)
	df_train_1['dis_similarity'] = pd.Series(np.ones(sample_size,)*1)

	# Remove null line
	print "Drop the null line..."
	# df_train_1 = df_train_1.dropna(subset=['en_article'])
	df_train_1 = df_train_1[df_train_1['en_article'] != '<NULL>']

	# Expand the training data
	en_article_wrong = df_train_1.en_article.iloc[random.sample(xrange(len(df_train_1)),len(df_train_1))]
	en_article_wrong.index = df_train_1.index
	print (en_article_wrong == df_train_1.en_article).value_counts()
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
		dir_en = "./data/news/en_news.csv"
		dir_jp = "./data/news/jp_news.csv"

		pairs_correct, pairs_wrong, df_pairs = prepare_train(dir_en, dir_jp)
		train_1 = pairs_correct[0:2000] + pairs_correct[3000:5000]
		test_1 = pairs_correct[2000:3000]

		train_2 = pairs_wrong[0:2000] + pairs_wrong[3000:5000]
	# test_2 = pairs_wrong[split_line:end_line]


	# Expand the training data
	train = train_1 + train_2


	# --- Apply the word2vec model to the data sets --- #

	model_name_en = "./data/model-en/W2Vmodle.bin"
	model_name_jp = "./data/model-jp/W2Vmodle.bin"

	df_pairs_sample = df_pairs.iloc[0:5000]

	df_pairs_sample['word2vec_en'] = df_pairs_sample['en_article'].apply(doc2vec_en)
	df_pairs_sample['word2vec_jp'] = df_pairs_sample['jp_article'].apply(doc2vec_jp)


	# Feature 1: TF-IDF + Average word2vec

	# --- Find tf-idf * word2vec features --- #

	#  For English text:
	texts_en = [doc.split() for doc in list(df_pairs_sample["en_article"])]
	texts_en_all = [doc.split() for doc in list(df_pairs["en_article"])]
	dictionary_en = corpora.Dictionary(texts_en_all)
	corpus_en = [dictionary_en.doc2bow(text) for text in texts_en]
	tfidf_en = models.TfidfModel(corpus_en)

	#  For Japanese text:
	texts_jp = [doc.split() for doc in list(df_pairs_sample["jp_article"])]
	texts_jp_all = [doc.split() for doc in list(df_pairs["jp_article"])]
	dictionary_jp = corpora.Dictionary(texts_jp_all)
	corpus_jp = [dictionary_jp.doc2bow(text) for text in texts_jp]
	tfidf_jp = models.TfidfModel(corpus_jp)

	features_en = doc2feature(corpus_en[:5000], tfidf_en, dictionary_en, model_en)
	features_jp = doc2feature(corpus_jp[:5000], tfidf_jp, dictionary_jp, model_jp)

	# --- When do not apply the tfidf re-weighting --- #
	flag_NO_tfidf = False
	if flag_NO_tfidf:

		df_pairs_sample["sum_vector_en"] = df_pairs_sample['word2vec_en'].apply(sum_docment)
		df_pairs_sample["sum_vector_jp"] = df_pairs_sample['word2vec_jp'].apply(sum_docment)
		df_pairs_sample["average_vector_en"] = df_pairs_sample['word2vec_en'].apply(average_docment)
		df_pairs_sample["average_vector_jp"] = df_pairs_sample['word2vec_jp'].apply(average_docment)

		features_en = list(df_pairs_sample["average_vector_en"])
		features_jp = list(df_pairs_sample["average_vector_jp"])


	features_merge = np.concatenate((features_en,features_jp), axis = 1)

	# --- Expanding the training data (dissimilar paris)
	features_en_wrong = np.array(features_en)
	np.random.shuffle((features_en_wrong))
	c = np.all(features_en_wrong == features_en, axis=1)
	print "C value =", c.sum() # check the duplicated amount

	features_merge_wrong = np.concatenate((features_en_wrong,features_jp), axis = 1)

	# --- Prepare the final training and test data --- #

	X = np.concatenate((features_merge, features_merge_wrong), axis = 0)
	y = np.concatenate((np.ones(len(features_merge)), np.zeros(len(features_en_wrong))), axis = 0)

	# --- Split into test data and training data --- #

	X_train1, X_test1, X_train2, X_train3_wrong, X_test2_wrong = np.split(X, [2000, 3000, 5000, 9000])
	y_train1, y_test1, y_train2, y_train3_wrong, y_test2_wrong = np.split(y, [2000, 3000, 9000, 9000])

	X_train = np.concatenate((X_train1, X_train2, X_train3_wrong), axis = 0)
	y_train = np.concatenate((y_train1, y_train2, y_train3_wrong), axis = 0)
	X_train_correct = np.concatenate((X_train1, X_train2), axis = 0)
	y_train_correct = np.concatenate((y_train1, y_train2), axis = 0)

	X_test = np.concatenate((X_test1, X_test2_wrong), axis = 0)
	y_test = np.concatenate((y_test1, y_test2_wrong), axis = 0)
	X_test_correct = X_test1
	y_test_correct = y_test1

	# --- SVM Training --- #

	clf = svm.SVC(kernel="rbf", gamma=0.001, C=100, probability=True)

	# 在使用
	standerlization = 0
	if standerlization == 1:
		print("Usinge the standardScaler")
		scaler = preprocessing.StandardScaler().fit(X_train)
		X_scaled = scaler.transform(X_train)
		X_test_scaled = scaler.transform(X_test)
		clf.fit(X_scaled, y_train)

		y_test_predict = clf.predict(X_test_scaled)
		y_train_predict = clf.predict(X_scaled)

	if standerlization == 2:
		print("Usinge the MinMaxScaler")
		min_max_scaler = preprocessing.MinMaxScaler()
		X_scaled = min_max_scaler.fit_transform(X_train)
		X_test_scaled = min_max_scaler.transform(X_test)
		clf.fit(X_scaled, y_train)

		y_test_predict = clf.predict(X_test_scaled)
		y_train_predict = clf.predict(X_scaled)
	else:
		print("Usinge the None")
		clf.fit(X_train, y_train)
		# clf.score(X_train, y_train)
		# clf.score(X_test, y_test)
		y_test_predict = clf.predict(X_test)
		y_train_predict = clf.predict(X_train)



	print "classification report of TRAINING data:"
	print(classification_report(y_train, y_train_predict))

	print "classification report of TEST data:"
	print(classification_report(y_test, y_test_predict))

	# Quick way
	q = wrapper_find_ranking_quick(X_test, clf)
	dic_rank_results_test_new = q.get()

	rank_results_test = [dic_rank_results_test_new[k] for k in sorted(dic_rank_results_test_new)]

	print(pd.Series(rank_results_test).describe())
	print("TOP1", (pd.Series(rank_results_test) <= 1).sum())
	print("TOP5", (pd.Series(rank_results_test) <= 5).sum())
	print("TOP10", (pd.Series(rank_results_test) <= 10).sum())

	# ---- Independent test data ---- #
	print("Using the new test data to evaluate.......")
	df_pairs_evaluate = df_pairs.iloc[50000:55000:5]

	texts_en_new = [doc.split() for doc in list(df_pairs_evaluate["en_article"])]
	corpus_en_new = [dictionary_en.doc2bow(text) for text in texts_en_new]

	texts_jp_new = [doc.split() for doc in list(df_pairs_evaluate["jp_article"])]
	corpus_jp_new = [dictionary_jp.doc2bow(text) for text in texts_jp_new]

	features_en_new = doc2feature(corpus_en_new[:1000], tfidf_en, dictionary_en, model_en)
	features_jp_new = doc2feature(corpus_jp_new[:1000], tfidf_jp, dictionary_jp, model_jp)

	features_merge_1_new = np.concatenate((features_en_new, features_jp_new), axis=1)

	# X_test_scaled_new = scaler.transform(features_merge_new)
	X_test_scaled_1_new = features_merge_1_new

	# sim_results_test, rank_results_test = find_ranking(X_test[:1000,:200] ,X_test[:1000,200:], clf)
	q = wrapper_find_ranking_quick(X_test_scaled_1_new, clf)
	dic_rank_results_test_new = q.get()

	rank_results_test_new = [dic_rank_results_test_new[k] for k in sorted(dic_rank_results_test_new)]

	print(pd.Series(rank_results_test_new).describe())
	print("TOP1", (pd.Series(rank_results_test_new) <= 1).sum())
	print("TOP5", (pd.Series(rank_results_test_new) <= 5).sum())
	print("TOP10", (pd.Series(rank_results_test_new) <= 10).sum())



	# --- Expanding the training data (dissimilar paris)
	features_en_wrong_new = np.array(features_en_new)
	np.random.shuffle((features_en_wrong_new))
	c = np.all(features_en_wrong_new == features_en_new, axis=1)
	print "C value =", c.sum() # check the duplicated amount

	features_merge_wrong_new = np.concatenate((features_en_wrong_new,features_jp_new), axis = 1)

	# --- Prepare the final training and test data --- #
	X_test_scaled_new = np.concatenate((features_merge_1_new, features_merge_wrong_new), axis = 0)

	y_test_predict_new = clf.predict(X_test_scaled_new)
	print "classification report of TEST data:"
	print(classification_report(y_test, y_test_predict_new))

"""

	# --- Prepare for a new independent evaluation balanced data --- #

	df_pairs_evaluate = df_pairs.iloc[55000:60000]

	df_pairs_evaluate['word2vec_en'] = df_pairs_evaluate['en_article'].apply(doc2vec_en)
	df_pairs_evaluate['word2vec_jp'] = df_pairs_evaluate['jp_article'].apply(doc2vec_jp)

	features_en_eva = doc2feature(corpus_en[60000:61000], tfidf_en, dictionary_en, model_en)
	features_jp_eva = doc2feature(corpus_jp[60000:61000], tfidf_jp, dictionary_jp, model_jp)

	features_merge_eva = np.concatenate((features_en_eva,features_jp_eva), axis = 1)

	features_en_wrong_eva =  features_en[:1000]
	# features_en_wrong_eva = np.array(features_en_eva)
	# np.random.shuffle((features_en_wrong_eva))
	# c = np.all(features_en_wrong_eva == features_en_eva, axis=1)
	# print "C value =", c.sum() # check the duplicated amount

	features_merge_wrong = np.concatenate((features_en_wrong_eva,features_jp_eva), axis = 1)

	X_eva = np.concatenate((features_merge_eva, features_merge_wrong), axis = 0)
	y_eva = np.concatenate((np.ones(len(features_merge_eva)), np.zeros(len(features_en_wrong_eva))), axis = 0)

	y_eva_predict = clf.predict(X_eva)

	print "classification report of TRAINING data:"
	print(classification_report(y_eva, y_eva_predict))


	# --- Evaluation for SVM --- #

	y_test_proba = clf.predict_proba(X_test)
	y_train_proba = clf.predict_proba(X_train)

	# sim_results_train, rank_results_train = find_ranking(projection1_train, projection2_train)
	sim_results_test, rank_results_test = find_ranking(X_test[:,:200] ,X_test[:,200:], clf)


	print pd.Series(rank_results_test).describe()

"""

