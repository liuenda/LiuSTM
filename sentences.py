import numpy
import numpy as np
import pickle
import random
# from nltk.corpus import stopwords
from gensim.models import word2vec

print "Loading Word2Vec"
data_path = "./data/"
# d2 = pickle.load(open(data_path + "synsem.p",'rb'))
dtr = pickle.load(open(data_path + "dwords.p", 'rb'))

maxlen = 100

# model = word2vec.Word2Vec.load_word2vec_format(data_path + "GoogleNews-vectors-negative300.bin.gz",binary = True)


# This is called by prepare_data() called by lstm.py
def getmtr(xa, maxlen):
    n_samples = len(xa)
    ls=[]
    x_mask = numpy.zeros((maxlen, n_samples)).astype(np.float32)
    for i in range(0, len(xa)):
        q = xa[i].split()
        for j in range(0, len(q)):
            x_mask[j][i] = 1.0
        while(len(q) < maxlen):
            q.append(',')
        ls.append(q)
    xa = np.array(ls)
    return xa, x_mask


# This is called lstm.py
# Here xa1 and xa2 are the sentence pair to be compared
# See onenote, search for xa1, xb2
# x1.shape,mas1.shape -> ((11219, 72), (72, 11219))
# In x1 and x2, all padding are replaced by commas: ','
def prepare_data(data):
    xa1 = []
    xb1 = []
    y2 = []
    for i in range(0,len(data)):
        xa1.append(data[i][0])
        xb1.append(data[i][1])
        #y2.append(round(data[i][2],0))
        y2.append(data[i][2])
    lengths = []
    for i in xa1:
        lengths.append(len(i.split()))
    for i in xb1:
        try:
            lengths.append(len(i.split()))
        except:
            print "the error is here", type(i), i
    maxlen = numpy.max(lengths)
    emb1, mas1 = getmtr(xa1, maxlen)
    emb2, mas2 = getmtr(xb1, maxlen)
    
    y2 = np.array(y2,dtype = np.float32)
    return emb1, mas1, emb2, mas2, y2


# This is called lstm.py
#new embed
def embed_old(stmx):
    #stmx=stmx.split()
    dmtr = numpy.zeros((stmx.shape[0], 300), dtype = np.float32)
    count = 0
    while(count < len(stmx)):
        if stmx[count] == ',':
            count += 1
            continue
        if stmx[count] in dtr:
            dmtr[count] = model[dtr[stmx[count]]]
            count += 1
        else:
            dmtr[count] = model[stmx[count]]
            count += 1
    return dmtr

def embed(stmx, k=10):
    dmtr = numpy.zeros((stmx.shape[0], k), dtype = np.float32)
    count = 0
    while(count < len(stmx)):
        if stmx[count] == ',':
            count += 1
            # print ","
            continue
        else:
            dmtr[count][int(stmx[count])-1] = 1
            # print int(stmx[count])-1
            count += 1
            # if int(stmx[count])-1 < 0 or int(stmx[count])-1 > 9:
            #     print "Error!!!!!!!"
    return dmtr
