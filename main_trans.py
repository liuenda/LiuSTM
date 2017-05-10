#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import time
from googletrans import Translator
# import pandas as pd
import MeCab
start_time = time.time()


import random
#import pickle
import time
import pandas as pd
import numpy as np
import lstm_trans as lstm

maxlen = 0 # Default: 0 -> infinite
epoch = 50
random.seed(1234)

def prepare_train(dir_en, dir_jp):

        df_en_mapping = pd.read_csv(dir_en)
        df_jp_mapping = pd.read_csv(dir_jp)

        #df_en_mapping = pd.read_table(dir_en, names=["en_article"])
        #df_jp_mapping = pd.read_table(dir_jp, names=["jp_article"])

        df_en_mapping = df_en_mapping.iloc[0:5000]
        df_jp_mapping = df_jp_mapping.iloc[0:5000]

        print "Reading english(translated) data:", len(df_en_mapping)
        print "Reading english Data:", len(df_jp_mapping)
        print df_en_mapping
        print df_jp_mapping

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

        print df_train_1
        # Convert dateframe to list
        train_1 = df_train_1[['en_article','jp_article','similarity']].values.tolist()
        train_2 = df_train_1[['en_article_wrong','jp_article','dis_similarity']].values.tolist()
        return train_1, train_2, df_train_1

"""
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
        except Exception as e:
                # info=''
                print e
                # counter_NaN+=1 #increase 1 to NaN counter
                # info+=repr(e)+"\n" #create log information
                # logout.write(info) #write log information to log file
                #new 3.15: generate a useless list for deleting in the next stage
                #output_unmatch.write(word) # no \n is needed since the
                #output_unmatch.write('\n')
                print "---Warning: Word ["+word+"] Vector Not Found ---"
                #return nan
                return None
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

def translate_document(input_filename, output_filename, n_document):

    text_list = []
    translate_text_list = []
    translator = Translator()
    output=open(output_filename,'w')
    with open(input_filename) as data_file:
        for (index,line) in enumerate(data_file):
            if index >= n_document:
                break
            else:
                text_list.append(line)

    for index, text in enumerate(text_list):
        try:
            translation = translator.translate(text[:1500],dest="ja")

        except Exception as e:
            print '=== エラー内容 ==='
            print 'type:' + str(type(e))
            print 'args:' + str(e.args)
            print 'message:' + e.message
            print 'e自身:' + str(e)
            translate_text_list.append("")
            output.write("[Failure]\n")

        else:
            translate_text_list.append(translation.text)
            output.write(translation.text.encode("utf_8")+"\n")

        time.sleep(0.5)
        print "Finish document No.", index
    output.close()

def tagging(input_filename, output_filename2):

    #input_filename=r'removed2_jp.csv'
    #output_filename1=r'tag_mecab_jp.txt'
    #output_filename2=r'cleaned_tag_jp.txt'

    #output1=open(output_filename1,'w')
    output2=open(output_filename2,'w')

    tagger = MeCab.Tagger("-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd/")


    with open(input_filename) as data_file:
            for (index,line) in enumerate(data_file):
                    #line=line.encode('utf-8','ignore')  # NO NEED!
                    node = tagger.parseToNode(line)
                    #index=0
                    line_tagged=[]
                    newLine=[]
                    while node:
                            word_tagged=(node.surface,node.feature)
                            line_tagged.append(word_tagged)
                            list_feature=node.feature.split(',')
                            if '動詞' in list_feature[0] or '名詞' in list_feature[0] or '接頭詞' in list_feature[0]:
                                    if '数' not in list_feature[1] and '接尾' not in list_feature[1]:
                                            if '*' not in list_feature[6]:
                                                    newLine.append(list_feature[6])
                            # if index==999:
                            # 	print list_feature[0]
                            node=node.next

                    output2.write(' '.join(newLine)+'\n')

                    if index in range(5000,60001,5000):
                            # print mecab_result+'\n\n'
                            print index

                    # output1.write('\n'.join('_'.join(t) for t in line_tagged))
                    # output1.write('\n\n\n')

                    # if index==999:
                    # 	print '\n'.join('_'.join(t) for t in line_tagged)
                    # # print index
    #output1.close()
    output2.close()


def clean_tag(input_filename, output_filename):
    #input_filename="cleaned_tag_jp.txt"
    #output_filename="cleaned2_tag_jp.txt"

    reg=[]
    reg.append(r'[ ]た[ ]*')   #When to use r'' When to use u''?
    reg.append(r'[ ]ない[ ]*')
    reg.append(r'[ ]だ[ ]*')

    output=open(output_filename,'w')
    with open(input_filename) as data_file:
            for (index,line) in enumerate(data_file):

                    if index in range(5000,60001,5000):
                            print "Now start the line No.:"+str(index)
                            print("--- %s seconds ---" % (time.time() - start_time))

                    #newData=line
                    #This must be run 1st! The order should not be changed!
                    for reg1 in reg:
                            line=re.sub(reg1,' ',line)

                    output.write(line)

    print("--- %s seconds ---" % (time.time() - start_time))
    output.close()


if __name__ == '__main__':

    flag_translate = False
    flag_tagging = False
    flag_txt2csv = True
    flag_training = 5000

    # input_filename = "../2016.5.11~Reuter/preprocessing/data/sample_removed2_en.csv"
    # output_filename = "./data/translation_en2jp_1000.txt"

    input_filename = "../2016.5.11~Reuter/preprocessing/data/removed2_en.csv"
    output_filename = "./data/translation_en2jp_5000.txt"

    n_document = 5000

    if flag_translate:
        translate_document(input_filename, output_filename, n_document)

    # Preprocessing the translated English document
    input_filename1 = "./data/translation_en2jp_5000.txt"
    output_filename1 = "tag_jp_trans5000.txt"

    input_filename2 = "tag_jp_trans5000.txt"
    output_filename2 = "./data_trans/cleaned_tag_jp_trans5000.txt"

    if flag_tagging:
        print "Start tagging the file:", input_filename1
        print "Output of the tagged file:", output_filename1
        tagging(input_filename1, output_filename1)
        print "Start to clean the tagging file:", input_filename2
        print "Output of the cleaned file:", output_filename2
        clean_tag(input_filename2, output_filename2)


   # Start data preparation and training
    k = 10

    if flag_txt2csv:
        # Add the null line with <NULL> mark
        output_filename3 = "./data_trans/wo_empty_cleaned_tag_jp_trans5000.txt"
        output=open(output_filename3,'w')
        with open(output_filename2, "r") as data_file:
            for (index,line) in enumerate(data_file):
                if line == "" or line == "\n":
                    print "find a null line in", output_filename2
                    line = '<NULL>\n'
                output.write(line)
        output.close()

        # Convert txt to csv files
        en = pd.read_table(output_filename3, names=["en_article"])
        print "length of the translated CSV data:", len(en), "[Expected 5000]"
        en.to_csv("./data_trans/cleaned_tag_jp_trans5000.csv")

    """
    if flag_training == 1000:
            # Prepare For the training data
            sample_size = "_1000"
            dir_en = "./data_trans/cleaned_tag_jp_trans5000.csv"
            dir_jp = "./data_trans/"

            # Prepare For the test data
            sample_size = "_1k2k"
            dir_en_test = "./data/mapping/en_mapped_"+str(k) + sample_size + ".csv"
            dir_jp_test = "./data/mapping/jp_mapped_" + str(k) + sample_size + ".csv"

            train_1, train_2, df_train_1 = prepare_train(dir_en, dir_jp)
            test_1, test_2, df_test_1 = prepare_train(dir_en_test, dir_jp_test)
    """

    if flag_training == 5000:
            # split_line = 5000
            # end_line = 6000
            # Prepare For the training data
            dir_en = "./data_trans/cleaned_tag_jp_trans5000.csv"
            dir_jp = "./jp_news.csv"

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
            dir_file = "weights/trans/20170413_e50_4000_b256.p"
            print "Starting to training the model..., saving to", dir_file
            sls=lstm.LSTM(dir_file, maxlen=maxlen, load=False, training=True)
            sls.train_lstm(train, epoch, train_1, test_1, batchsize=batchsize)
            sls.save_model()
    else:
            dir_file = "weights/trans/20170413_e50_4000_b256.p"
            print "NO Training. Load the existed model:", dir_file
            sls=lstm.LSTM(dir_file, maxlen=maxlen, load=True, training=False)

    if True:
            print "Evaluate the model using fast estimation..."
            projection1_train, projection2_train = sls.seq2vec(train_1)
            projection1_test, projection2_test = sls.seq2vec(test_1)

            sim_results_train, rank_results_train = lstm.find_ranking(projection1_train, projection2_train)
            sim_results_test, rank_results_test = lstm.find_ranking(projection1_test, projection2_test)

            print pd.Series(rank_results_train).describe()
            print pd.Series(rank_results_test).describe()
