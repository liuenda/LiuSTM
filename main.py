# coding: utf-8
import pandas as pd
from gensim.models import *
import numpy as np
import time
from ast import literal_eval
import lstm
import random
import pickle

# TODO1: 文件输入en，jp的txt文件的时候，需要检查1. 是否行数一直 2. 是否有\n\n的问题出现
# 目前是手动删除对应的\n\n行[已完成]

# TODO1: 每次都要计算imilarity table效率太低了。1000行还可接受，但是更多行就不行了
# 首先边计算边存储到一个dictionary去[已完成]

epoch = 40
maxlen = 0 # Default: 0 -> infinite
k = 10
wnl = 0
dim = 200
nan = np.empty(dim)
counter = 0 
start_time = time.time()
dic_mapping = {}
# np.random.seed(1234)
random.seed(1234)

dir_cluster_center = './data/cluster-skmeans/'
model_name_en = "./data/model-en/W2Vmodle.bin"
model_name_jp = "./data/model-jp/W2Vmodle.bin"
dir_mapping = "./data/mapping/mapping_en_" + str(k) + ".csv"

log_filename1 = "./log/output_unmatch_jp.log"
output_unmatch = open(log_filename1,'w')

if False:
	# # For ALL
	sample_size = ""
	dir_txt_en = "./data/news/wo_empty_line_en" + sample_size + ".txt"
	dir_txt_jp = "./data/news/wo_empty_line_jp" + sample_size + ".txt"

else:
	# #For sample
	# #sample size
	sample_size = "_1000"
	dir_txt_en = "./data/news/wo_empty_line_en" + sample_size + ".txt"
	dir_txt_jp = "./data/news/wo_empty_line_jp" + sample_size + ".txt"



# Call mapping_word
def mapping_article(article,model):
	# start_time = time.time()
	global counter
	print counter
	counter = counter + 1
	tokens = article.split()
	tokens_mapping=[mapping_word(word,model) for word in tokens]
	# print "DEBUG: Finish 1 line-------------"
	# print time.time() - start_time
	return tokens_mapping

# Call: get_vector
# Find the nearest en-cluster for a given word
def mapping_word(word,model):
	# 1. get the center for each en-cluster
	df_center_en=find_cluster_center(dir_cluster_center,'en')
	
	if word in dic_mapping:
		cluster_number = dic_mapping[word]
		# print "IN!"
		return cluster_number
	else:
		# 2. find the word2vec expression
		vec=get_vector(word,model)
		# print "OUT----"
		# print "DEBUG: ",vec
		
		if np.all(vec!=nan): 
			# 3. calculate the similarity matrix
			similarity_matrix_en = \
				np.array(df_center_en).dot(vec) # ????? Have a check!
			# print "DEBUG, similarity_matrix_en ="
			# print "with shape of ", np.shape(similarity_matrix_en)
			
			# 4. Get the maximum one that can present this cluster
			cluster_number=similarity_matrix_en.argmax()+1

			# 5. add the mapping to the dictionary
			dic_mapping[word] = cluster_number

			return cluster_number
		else:
			print "Error: vec == NaN"
			return None


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

# Find the center of each cluster
def find_cluster_center(cluster_centroid_dir,lang_name):
	cluster_center_filename=cluster_centroid_dir + "centroid_" + lang_name + str(k) + ".csv"
	df_cluster_center = pd.read_csv(cluster_center_filename,index_col=0)
	# print "DEBUG: df_cluster_center is [" +  cluster_center_filename + "]" 
	# print df_cluster_center
	return df_cluster_center


def map_to_jp_vector(vector_en, df_mapping):
	result = []
	for cluster_name_en in vector_en:
		if cluster_name_en != None:
			cluster_name_converted = df_mapping.iloc[cluster_name_en-1].mapping_parsed
			result += list(cluster_name_converted)
		else:
			result = None
	# print result
	return result


def evaluate_1(k):
	xa = df_train_1['xa'][k]
	return evaluate(xa)

def evaluate(xa):
	xa_0_result = df_train_1['xb'].apply(sls.predict_similarity,args=(xa,))
	ranking = xa_0_result.rank(ascending = False)[k]
	# print ranking
	return ranking


def evaluate_all(df):
	df_result = df.xa.apply(evaluate)
	return df_result


# -----------------------Prepare the mapped Data----------------

def prepare_trainig(dir_en, dir_jp):

	# Read the saved mapping results:
	df_en_mapping = pd.read_csv(dir_en)
	df_jp_mapping = pd.read_csv(dir_jp)

	print "Reading english Data:", len(df_en_mapping)
	print "Reading english Data:", len(df_jp_mapping)

	sample_size = len(df_en_mapping)

	assert len(df_en_mapping) == len(df_jp_mapping)

	# Convert mapping to list type and then concat to the a list
	print "Merging the English and Japanes news dataframe"
	df_train_1 = pd.concat([df_en_mapping, df_jp_mapping], axis = 1)
	df_train_1['transformation_en'] = df_train_1.transformation_en.apply(literal_eval)
	df_train_1['transformation_jp'] = df_train_1.transformation_jp.apply(literal_eval)
	# df_train_1['similarity'] = pd.Series(np.ones(int(sample_size[1:]),))
	df_train_1['similarity'] = pd.Series(np.ones(sample_size,)*5)
	df_train_1['dis_similarity'] = pd.Series(np.ones(sample_size,)*1)



	# prepare the multi-lingual cluster mapping
	print "Read mapping file and convert it to tuple from string"
	df_mapping = pd.read_csv(dir_mapping)
	df_mapping['mapping_parsed'] = df_mapping.mapping.map(lambda x: literal_eval(x)) 

	# Call map_to_jp_vector()
	print "Mapping English clusters to Japanese clusters"
	df_train_1['en2jp_projection'] = \
		df_train_1['transformation_en'].apply(map_to_jp_vector,args=(df_mapping,))

	# Remove null line
	print "Drop the null line"
	df_train_1 = df_train_1.dropna(subset=['en2jp_projection'])

	# Convert list of cluster number to a string 
	print "Convert cluster names(list) to cluster namse(string)"
	df_train_1[['xa','xb']] = df_train_1[['transformation_en','en2jp_projection']].applymap(lambda x:' '.join(str(v) for v in x))


	# Expand the training data
	xb_wrong = df_train_1.xb.iloc[random.sample(xrange(len(df_train_1)),len(df_train_1))]
	xb_wrong.index = df_train_1.index
	print (xb_wrong == df_train_1.xb).value_counts()
	df_train_1['xb_wrong'] = xb_wrong


	# Convert dateframe to list
	train_1 = df_train_1[['xa','xb','similarity']].values.tolist()
	train_2 = df_train_1[['xa','xb_wrong','dis_similarity']].values.tolist()

	return train_1, train_2, df_train_1

# # Called by find_ranking
# # Given 2 list of projection results, calculate there L1-norm similarity
# def cal_similarity(a, b):
# 	diff = np.linalg.norm(a - b, 1, axis=1)
# 	sim = np.exp(-diff)
# 	# len(diff)
# 	return sim

# # Find the ranking results with respect to real pairs
# def find_ranking(projection1, projection2):
# 	sim_results = []
# 	rank_results = []
# 	for i, proj1 in enumerate(projection1):
# 		sim = cal_similarity(proj1, projection2)
# 		rank = pd.Series(sim).rank(ascending = False)[i]
# 		sim_results.append(sim)
# 		rank_results.append(rank)
# 	return sim_results, rank_results


if __name__ == "__main__":

	#-----------------------------Loading-------------------------

	model_en = Word2Vec.load(model_name_en)
	model_jp = Word2Vec.load(model_name_jp)

	# -----------------------Mapping Raw News Data----------------
	# Read news data
	df_en = pd.read_table(dir_txt_en, names=["en_article"])
	df_jp = pd.read_table(dir_txt_jp, names=["jp_article"])

	# Mapping cluster name For Enlgish news
	# and save the file 
	if False:
		print "Mapping cluster name For Enlgish news"
		start_time = time.time()
		df_en['transformation_en'] = \
			df_en.en_article.apply(mapping_article,args=(model_en,))
		df_en.to_csv("./data/mapping/en_mapped_" + str(k) + sample_size + ".csv",index=False)
		print time.time() - start_time

	# Mapping cluster name For Japanese news
	# and save the file
	if False:
		print "Mapping cluster name For Japanese news"
		start_time = time.time()
		df_jp['transformation_jp'] = \
			df_jp.jp_article.apply(mapping_article,args=(model_jp,))
		df_jp.to_csv("./data/mapping/jp_mapped_" + str(k) + sample_size + ".csv",index=False)
		print time.time() - start_time

	# -----------------Formatting the data------------------------

	if False:
		# Prepare For the training data
		dir_en = "./data/mapping/en_mapped_"+str(k) + sample_size + ".csv"
		dir_jp = "./data/mapping/jp_mapped_" + str(k) + sample_size + ".csv"
		train_1, train_2, df_train_1 = prepare_trainig(dir_en, dir_jp)

		# Prepare For the testing data
		sample_size = "_1k2k"
		dir_en = "./data/mapping/en_mapped_"+str(k) + sample_size + ".csv"
		dir_jp = "./data/mapping/jp_mapped_" + str(k) + sample_size + ".csv"
		test_1, test_2, df_test_1 = prepare_trainig(dir_en, dir_jp)

		# ----save the prepared data into pickle-----------------------
		root_dir = "pickles/"
		with open(root_dir + "train_1.p", 'wb') as handle:
		    pickle.dump(train_1, handle)
		with open(root_dir + "train_2.p", 'wb') as handle:
		    pickle.dump(train_2, handle)
		with open(root_dir + "test_1.p", 'wb') as handle:
		    pickle.dump(test_1, handle)
		with open(root_dir + "test_2.p", 'wb') as handle:
		    pickle.dump(test_2, handle)
		with open(root_dir + "df_train_1.p", 'wb') as handle:
		    pickle.dump(df_train_1, handle)
		with open(root_dir + "df_test_1.p", 'wb') as handle:
		    pickle.dump(df_test_1, handle)

	else:
		root_dir = "pickles/"
		# ------load the exited prepared data from pickle---------------
		train_1 = pickle.load(open(root_dir + "train_1.p",'rb'))
		train_2 = pickle.load(open(root_dir + "train_2.p",'rb'))
		test_1 = pickle.load(open(root_dir + "test_1.p",'rb'))
		# test_2 = pickle.load(open(root_dir + "test_2.p",'rb')) 
		df_train_1 = pickle.load(open(root_dir + "df_train_1.p",'rb'))
		df_test_1 = pickle.load(open(root_dir + "df_test_1.p",'rb'))



	#---------------------- 1 time training ------------------------
	#-----------------------Load/Train the LSTM model---------------

	train = train_1 + train_2

	# True to training the data, False to laod the existed data
	print "Now the maxlen =", maxlen
	if True:
		dir_file = "weights/2017012218009_e40_1k1k_l0.p"
		print "Starting to training the model..., saving to", dir_file
		sls=lstm.LSTM(dir_file, maxlen, load=False, training=True)
		sls.train_lstm(train, epoch, train_1, test_1)
		sls.save_model()
	else:
		dir_file = "weights/201701102308_e40_1k1k_l0.p"
		print "NO Training. Load the existed model:", dir_file
		sls=lstm.LSTM(dir_file, maxlen, load=True, training=False)


	#--- New method to evaluate the results ------------------------
	#--------------------Evaluate the results using new method------
	if False:
		print "Evaluate the model using fast estimation..."
		projection1_train, projection2_train = sls.seq2vec(train_1)
		projection1_test, projection2_test = sls.seq2vec(test_1)

		sim_results_train, rank_results_train = lstm.find_ranking(projection1_train, projection2_train)
		sim_results_test, rank_results_test = lstm.find_ranking(projection1_test, projection2_test)

		print pd.Series(rank_results_train).describe()
		print pd.Series(rank_results_test).describe()

		## Save the training results to pickle
		# root_dir = "pickles/"
		# with open(root_dir + "rank_results_train_20161214.py", 'wb') as handle:
		#     pickle.dump(train_1, handle)
		# with open(root_dir + "train_1.p", 'wb') as handle:
		#     pickle.dump(train_1, handle)

	#----- multiple time trainings to find optimal parameters -------

	# mse_maxlen = {}
	# mse_maxlen_train = {}
	# mse_maxlen_test = {}
	# time_cost = {}

	# def save_mse_maxlen(maxlen, sls):
	# 	mse_maxlen[maxlen] = list(sls.mse)
	# 	mse_maxlen_train[maxlen] = list(sls.mse_train)
	# 	mse_maxlen_test[maxlen] = list(sls.mse_test)
	# 	time_cost[maxlen] = sls.time_saver

	# for length in range(50,1000,50):
	# 	maxlen = length
	# 	dir_file = "weights/e40_1k1k_l.p" + str(maxlen) + ".p"
	# 	print "Starting to training the model..., saving to", dir_file
	# 	sls = lstm.LSTM(dir_file, maxlen, load=False, training=True)
	# 	sls.train_lstm(train, epoch, train_1, test_1)
	# 	save_mse_maxlen(maxlen, sls)
	# 	with open(root_dir + "mse_maxlen.p", 'wb') as handle:
	# 		pickle.dump(mse_maxlen, handle)
	# 	with open(root_dir + "mse_maxlen_train.p", 'wb') as handle:
	# 		pickle.dump(mse_maxlen_train, handle)
	# 	with open(root_dir + "mse_maxlen_test.p", 'wb') as handle:
	# 		pickle.dump(mse_maxlen_test, handle)
	# 	with open(root_dir + "time_cost.p", 'wb') as handle:
	# 		pickle.dump(time_cost, handle)


