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
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import random
import pickle
from sklearn import preprocessing
import time


model_name_en = "./data/model-en/W2Vmodle.bin"
model_name_jp = "./data/model-jp/W2Vmodle.bin"

model_en = word2vec.Word2Vec.load(model_name_en)
model_jp = word2vec.Word2Vec.load(model_name_jp)


maxlen = 0 # Default: 0 -> infinite
epoch = 50
random.seed(1234)


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


	# --- Train or load the doc2vec model --- #
	flag_train_doc2vec = False

	if flag_train_doc2vec:

		# --- Prepare training data of doc2vec --- #
		doc2vec_corpus_en = []
		doc2vec_corpus_jp = []
		for i, doc in enumerate(pairs_correct):
			doc2vec_corpus_en.append(TaggedDocument(words=doc[0].split(), tags=[i]))
			doc2vec_corpus_jp.append(TaggedDocument(words=doc[1].split(), tags=[i]))

		# --- Train the doc2vec model --- #
		doc2vec_model_en = doc2vec.Doc2Vec(doc2vec_corpus_en, size=200, window=8, min_count=1, workers=14)
		doc2vec_model_jp = doc2vec.Doc2Vec(doc2vec_corpus_jp, size=200, window=8, min_count=1, workers=14)

		# --- Save the doc2vec model --- #
		doc2vec_model_jp.save("./data/doc2vec_model_jp")
		doc2vec_model_en.save("./data/doc2vec_model_en")

	else:
		# --- Load the saved doc2vec model --- #
		doc2vec_model_en = doc2vec.Doc2Vec.load("./data/doc2vec_model_jp")
		doc2vec_model_jp = doc2vec.Doc2Vec.load("./data/doc2vec_model_en")


	# --- Evaluation 1: Using the cross-lingual projection directly --- #

	# --- Evaluation 2: Using SVM training --- #

	features_en = list(doc2vec_model_en.docvecs)[:5000]
	features_jp = list(doc2vec_model_jp.docvecs)[:5000]
	features_merge = np.concatenate((features_en,features_jp), axis = 1)

	# --- Expanding the training data (dissimilar paris)
	features_en_wrong = np.array(features_en)
	np.random.shuffle(features_en_wrong)
	c = np.all(features_en_wrong == features_en, axis=1)
	print "C value =", c.sum() # check the duplicated amount

	features_merge_wrong = np.concatenate((features_en_wrong,features_jp), axis = 1)

	# --- Prepare the final training and test data --- #

	X = np.concatenate((features_merge, features_merge_wrong), axis = 0)
	y = np.concatenate((np.ones(len(features_merge)), np.zeros(len(features_en_wrong))), axis = 0)

	# --- Split into test data and training data --- #

	X_train1, X_test, X_train2, X_train3_wrong = np.split(X, [2000, 3000, 5000])
	y_train1, y_test, y_train2, y_train3_wrong = np.split(y, [2000, 3000, 5000])

	X_train = np.concatenate((X_train1, X_train2, X_train3_wrong), axis = 0)
	y_train = np.concatenate((y_train1, y_train2, y_train3_wrong), axis = 0)
	X_train_correct = np.concatenate((X_train1, X_train2), axis = 0)
	y_train_correct = np.concatenate((y_train1, y_train2), axis = 0)

	# --- SVM Training --- #
	start = time.clock()
	# clf = svm.SVC()
	clf = svm.SVC(kernel="rbf", gamma=0.001, C=1, probability=True)


	# --- 归一化数据 --- #
	standerlization = 1
	if standerlization == 1:
		scaler = preprocessing.StandardScaler().fit(X_train)
		X_scaled = scaler.transform(X_train)
		X_test_scaled = scaler.transform(X_test)
		clf.fit(X_scaled, y_train)

		y_test_predict = clf.predict(X_test_scaled)
		y_train_predict = clf.predict(X_scaled)

	if standerlization == 2:
		min_max_scaler = preprocessing.MinMaxScaler()
		X_scaled = min_max_scaler.fit_transform(X_train)
		X_test_scaled = min_max_scaler.transform(X_test)
		clf.fit(X_scaled, y_train)

		y_test_predict = clf.predict(X_test_scaled)
		y_train_predict = clf.predict(X_scaled)
	else:
		clf.fit(X_train, y_train)
		# clf.score(X_train, y_train)
		# clf.score(X_test, y_test)
		y_test_predict = clf.predict(X_test)
		y_train_predict = clf.predict(X_train)


	# clf = svm.SVC(kernel="linear", probability=True)
	# clf.fit(X_train, y_train)
	print "Time cost for SVC fitting is", time.clock() - start
	# clf.score(X_train, y_train)
	# clf.score(X_test, y_test)

	# y_test_predict = clf.predict(X_test)
	# y_train_predict = clf.predict(X_train)

	print "classification report of TRAINING data:"
	print(classification_report(y_train, y_train_predict))

	print "classification report of TEST data:"
	print(classification_report(y_test, y_test_predict))

	y_test_proba = clf.predict_proba(X_test)
	y_train_proba = clf.predict_proba(X_train)

	# --- Evaluation for SVM --- #

	start = time.clock()
	# sim_results_train, rank_results_train = find_ranking(projection1_train, projection2_train)
	sim_results_test, rank_results_test = find_ranking(X_test[:,:200] ,X_test[:,200:], clf)
	print "Time cost for finding ranking", time.clock() - start

	print pd.Series(rank_results_test).describe()
