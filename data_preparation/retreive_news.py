# coding: utf-8

"""
created on 2017/06/02
@author: liuenda
"""

import numpy as np
import pandas as pd
import glob
import MeCab
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet as wn

base_path = "/home/M2015eliu/Reuters_News_download/2014/"
frame = pd.DataFrame()
list_jp = []
list_merge = []
list_en = []
month_start = 1
month_end =  12
year = 2014
period = ["%.2d" % i for i in range(month_start, month_end+1)]

column_need = ["PNAC", "UNIQUE_STORY_INDEX", "HEADLINE_ALERT_TEXT", "TAKE_TEXT",
               "reference_code"]
column_need_JP = ["PNAC_JP", "UNIQUE_STORY_INDEX_JP", "HEADLINE_ALERT_TEXT_JP",
                  "TAKE_TEXT_JP"]
column_need_EN = ["PNAC_EN", "UNIQUE_STORY_INDEX_EN", "HEADLINE_ALERT_TEXT_EN",
                  "TAKE_TEXT_EN"]


work_path = "/home/M2015eliu/cas/2017.1.1~LiuSTM/data_preparation/"
file_name = "2014_merge_jp_en.csv"
file_name2 = "2014_cleaned_jp_en.csv"

retrieve_from_db = False
preprocessing = True

# --- Load CSV file with JP and EN filtering --- #
if retrieve_from_db:
    for month in period:
        # Load the file monthly
        file_path = base_path + "rna002_RTRS_" + str(year) + "_" + month + ".csv"
        df = pd.read_csv(file_path, index_col=None, header=0)

        # Filter out the Japanese articles
        df_month_jp = df[df["LANGUAGE"].isin(["JA"])
                         & df["EVENT_TYPE"].isin(["STORY_TAKE_OVERWRITE"])
                         & df["TAKE_TEXT"].str.contains("参照番号")]

        # Filter out the English articles
        df_month_en = df[df["LANGUAGE"].isin(["EN"])
                         & df["EVENT_TYPE"].isin(["STORY_TAKE_OVERWRITE"])]

        # Extract the reference code for Japanese news [会有警告，为何？]
        df_month_jp["reference_code"] = df_month_jp["TAKE_TEXT"].str.extract('参照番号\\[([\\w]+)\\]')

        # Drop the row where reference code is NaN
        df_month_jp = df_month_jp.dropna(subset=["reference_code"])

        # Find the English news basing on the Japanese news
        df_month_merge = pd.merge(df_month_jp[column_need].reset_index(),
                                  df_month_en[column_need[:-1]].reset_index(),
                                  left_on='reference_code',
                                  right_on='PNAC',
                                  how="left").dropna()
        list_merge.append(df_month_merge)

        # # Append the Data frame to the list
        # list_jp.append(df_month_jp[column_need])
        # list_en.append(df_month_en[column_need[:-1]])

        print "[I] Finish referenece extraction for month", month

    # Merge to one dataframe
    df_merge = pd.concat(list_merge)

    # Save to a csv file
    df_merge.to_csv(work_path + file_name)

else:
    df_merge = pd.read_csv(work_path + file_name, index_col=0).reset_index()


# ---- Pre-processing ---- #

def tagging_jp(text_jp):

    tagger = MeCab.Tagger("-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd/")

    node = tagger.parseToNode(text_jp)
    line_tagged=[]
    newLine = []
    while node:
        word_tagged = (node.surface, node.feature)
        line_tagged.append(word_tagged)
        list_feature = node.feature.split(',')
        if '動詞' in list_feature[0] or '名詞' in list_feature[0] or '接頭詞' in list_feature[0]:
            if '数' not in list_feature[1] and '接尾' not in list_feature[1]:
                if '*' not in list_feature[6]:
                    newLine.append(list_feature[6])
        node = node.next

    text_tagged_jp = ' '.join(newLine)

    return text_tagged_jp


def clean_tag_jp(tagged_text_jp):

    reg=[]
    reg.append(r'[ ]た[ ]*')   #When to use r'' When to use u''?
    reg.append(r'[ ]ない[ ]*')
    reg.append(r'[ ]だ[ ]*')

    for reg1 in reg:
        tagged_text_jp = re.sub(reg1, ' ', tagged_text_jp)

    return tagged_text_jp

def tagging_en(text_en):

    # Before tagging, remove unneeded parts
    text_en = text_en.replace("-", " ")
    text_en = text_en.replace("=", " ")

    tagger = nltk.PerceptronTagger()
    text_en_tagged = ""
    text_en = text_en.decode('utf-8','ignore') # you have to decode the line using the corresponded coding!

    word_list = nltk.word_tokenize(text_en)
    line_tagged = tagger.tag(word_list)

    for t in line_tagged:
        text_en_tagged += ('_'.join(t)+' ')
    return text_en_tagged

def clean_tag_en(tagged_text_en):

    wnl = WordNetLemmatizer()

    # the reference http://www.comp.leeds.ac.uk/amalgam/tagsets/upenn.html
    reg = []
    reg.append(r'[^ ]+_CD')  # mid-1890 nine-thirty forty-two one-tenth ten million 0.5
    reg.append(r'[^ ]+_DT')  # all an another any both del each either every half la many
    reg.append(r'[^ ]+_EX')  # there
    reg.append(r'[^ ]+_CC')  # & 'n and both but either et for less minus neither nor or plus so
    reg.append(r'[^ ]+_IN')  # astride among uppon whether out inside pro despite on by throughou
    reg.append(r'[^ ]+_SYM')  # % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R \* \*\* \*\*\*
    reg.append(r'[^ ]+_RP')  # aboard about across along apart around aside at away back before
    reg.append(r'[^ ]+_TO')  # to

    for reg1 in reg:
        tagged_text_en = re.sub(reg1,'',tagged_text_en)

    wordList = tagged_text_en.lower().split()

    finalList = []
    for i, w in enumerate(wordList):
        # ADD 16/1/26 Transform ADV to ADJ
        # ADD 16/1/28 ('_JJ' in w) to resovle words like "only_JJ"
        # if ('_RB' in w) or ('_JJ' in w):
        if ('_rb' in w):
            advset = w[:w.find('_')] + ".r.1"
            try:
                adj = wn.synset(advset).lemmas()[0].pertainyms()[0].name()
                w = w.replace(w, adj + '_jjr')
            except (IndexError, WordNetError):
                w = w.replace(w, w[:w.find('_')] + '_jjr')

        if ('_jjr' in w) or ('_jjs' in w):
            # newADJ=wnl.lemmatize(w[:-4], 'a')
            newADJ = wnl.lemmatize(w[:w.find('_')], 'a')
            w = w.replace(w, newADJ + '_jj')
        # print "JJR replacement,the NewList:",w,"To",newADJ

        # HERE the ('_nn' in w) is to remedy the ERROR of Tagging('weaker_NN')
        if '_nn' in w:
            old = w[:w.find('_')]
            newADJ = wnl.lemmatize(w[:w.find('_')], 'a')
            w = w.replace(w, newADJ + '_nn')
            if old != newADJ:
                print "NN--ADJ error: " + old + " " + newADJ

        # CODE:W1
        # Here is a big hazard, since _p can refer to '_pos'!!which will also be converted to nn!!
        # PDT Predeterminer POS Possessive ending PRP Personal pronoun PRP$ Possessive pronoun
        # convert 'its' to 'it'
        if ('_nn' in w or '_pr' in w):
            newNoun = wnl.lemmatize(w[:w.find('_')], 'n')
            w = w.replace(w, newNoun + '_nn')

        if ('_v' in w):
            newNoun = wnl.lemmatize(w[:w.find('_')], 'v')
            w = w.replace(w, newNoun + '_vb')


        finalList.append(w)

    # Re-combine into a string and remove all the POS-tags
    newLine = " ".join(finalList)

    # Output for normal tagging
    newLine1 = newLine
    # Remove punctuations and other symbols
    newLine1 = re.sub(r'[^ \n]+_[^A-Za-z \n]+', '', newLine1)
    # Remove unknown nouns xxx_nn
    newLine1 = re.sub(r'[^ ]*[^A-Za-z.]_[Nn][^ \n]+', '',
                      newLine1)  # Here is the reason that the ' are deleted!!! CODE:W1
    # Remove all the tagger notatation "_xxx"
    newLine1 = re.sub(r'_[^ \n]+', '', newLine1)
    # Reshape the string by removing continuous space
    newLine1 = re.sub(r' [ ]+', ' ', newLine1)

    # Remove 's
    newLine1 = newLine1.replace(" 's", "")

    # Remove '
    newLine1 = newLine1.replace("'", "")

    return newLine1

if preprocessing:

    df_title = df_merge[["HEADLINE_ALERT_TEXT_x","HEADLINE_ALERT_TEXT_y"]]

    # --- Pre-processing for Japanese --- #
    df_title["head_tagged_jp"] = df_title["HEADLINE_ALERT_TEXT_x"].apply(tagging_jp)
    df_title["head_tagged2_jp"] = df_title["head_tagged_jp"].apply(clean_tag_jp)

    # --- Pre-processing for English --- #
    df_title["head_tagged_en"] = df_title["HEADLINE_ALERT_TEXT_y"].apply(tagging_en)
    df_title["head_tagged2_en"] = df_title["head_tagged_en"].apply(clean_tag_en)


    df_title[["head_tagged2_en", "head_tagged2_jp"]].to_csv(work_path + file_name2, encoding = "utf-8")
    # df_title.to_json("test.json")
    #
    # df = pd.read_json(df_title)
