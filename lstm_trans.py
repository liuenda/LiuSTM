# coding: utf-8
import pickle
import numpy as np
import scipy.stats as meas
from collections import OrderedDict
import time
import theano
from theano import config
import theano.tensor as T
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# from sentences import *
# from random import sample
import random
from sentences_trans import prepare_data as prepare_data
from sentences_trans import embed_w2v as embed
import pandas as pd
# from sentences import expand as expand # Useless in the later tasks

# data_path = "./data/"
# test = pickle.load(open(data_path + "semtest.p",'rb')) # add by liuenda

options = locals().copy()

random.seed(1234)
np.random.seed(1234)

# Called by find_ranking
# Given 2 list of projection results, calculate there L1-norm similarity
def cal_similarity(a, b):
    diff = np.linalg.norm(a - b, 1, axis=1)
    sim = np.exp(-diff)
    # len(diff)
    return sim

# Find the ranking results with respect to real pairs
# Defaulty, projection1 should be JP
# Whiile, projection2 should be EN->JP
def find_ranking(projection1, projection2):
    sim_results = []
    rank_results = []

    # Iterate each of the ariticle from projection1 (999) as proj1
    # Calculate the simialrity of proj1 with all ariticles in projection2 (999)
    for i, proj1 in enumerate(projection1):
        sim = cal_similarity(proj1, projection2)
        rank = pd.Series(sim).rank(ascending = False)[i]
        sim_results.append(sim)
        rank_results.append(rank)

    # sim_results contains 999*999 similairty matrix
    return sim_results, rank_results

# rank_results should be list of (999,)
def find_top(rank_results, top):
    s = pd.Series(rank_results)
    n_top = (s <= top).sum()
    return n_top

# ---------------------------------------------

# Make a new name
# combine pp and name to "pp_name"
def _p(pp, name):
    return '%s_%s' % (pp, name)

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

use_noise = theano.shared(numpy_floatX(0.))

# NOT used
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# Create Tensor Shared variable(parameters) from Dicitonary of Weights (WUb)
# and then save the tensors into a new Dictionary tparams
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name = kk)
    return tparams

def get_layer(name):
    fns = layers[name]
    return fns

# 生成一个正态分布随机矩阵
# Generate random numbers according to standard deviation
# mu: mean, sigma: deviation
# n1, n2: size of the matrix
def genm(mu, sigma, n1, n2):
    return np.random.normal(mu, sigma, (n1, n2))

# 生成一个LSTM单元，参数U，W，b三组，并且初始化
# Example: newp = getlayerx(newp, '1lstm1', 50, 300)
# d: OrderedDictionary, pref: prefix (name), n: timesteps, nin: input dimension
def getlayerx(d, pref, n, nin):

    # mean value for normal distribution
    mu = 0.0

    # deviation for normal distribution
    sigma = 0.2

    # U, with random initialization
    U = np.concatenate([genm(mu, sigma, n, n), genm(mu, sigma, n, n), genm(mu, sigma, n, n), genm(mu, sigma, n, n)]) / np.sqrt(n)
    U = np.array(U, dtype = np.float32)

    # W, with random initialization
    W = np.concatenate([genm(mu, sigma, n, nin), genm(mu, sigma, n, nin), genm(mu, sigma, n, nin), genm(mu, sigma, n, nin)]) / np.sqrt(np.sqrt(n * nin))
    W = np.array(W, dtype = np.float32)

    # b, with random initialization
    # Initialize the b_i, b_f, b_c and b_o in the same time
    b = np.random.uniform(-0.5, 0.5, size=(4*n,))

    # set thhe bias of the forget gates b[n:n+n]) to 1.5
    #b = numpy.zeros((n * 300,))+1.5
    b[n:n*2] = 1.5

    # Update the dictionary
    d[_p(pref, 'U')] = U
    d[_p(pref, 'W')] = W
    d[_p(pref, 'b')] = b.astype(config.floatX)

    return d

# Here the hidden unite is set to be 50
def creatrnnx():
    newp = OrderedDict()
    #print ("Creating neural network")
    newp = getlayerx(newp, '1lstm1', 50, 200)
    #newp=getlayerx(newp,'1lstm2',30,50)
    #newp=getlayerx(newp,'1lstm3',40,60)
    #newp=getlayerx(newp,'1lstm4',6)
    #newp=getlayerx(newp,'1lstm5',4)
    newp = getlayerx(newp, '2lstm1', 50, 200)
    #newp=getlayerx(newp,'2lstm2',20,10)
    #newp=getlayerx(newp,'2lstm3',10,20)
    #newp=getlayerx(newp,'2lstm4',6)
    #newp=getlayerx(newp,'2lstm5',4)
    #newp=getlayerx(newp,'2lstm3',4)
    #newp['2lstm1']=newp['1lstm1']
    #newp['2lstm2']=newp['1lstm2']
    #newp['2lstm3']=newp['1lstm3']
    return newp

def dropout_layer(state_before, use_noise, rrng,rate):
    proj = tensor.switch(use_noise,
                         (state_before *rrng),
                         state_before * (1-rate))
    return proj


# Usage example:
# proj11 = getpl2(prevlayer =emb11, pre = '1lstm1', mymask = mask11,
#                   used = False, rrng, size = 50, tnewp)[-1]
def getpl2(prevlayer, pre, mymask, used, rrng, size, tnewp):
	# lstm_layer2() returns the value of hidden layer (hvals)
    proj = lstm_layer2(tnewp, prevlayer, options,
                                        prefix=pre,
                                        mask=mymask,nhd=size)
    if used:
        print "Added dropout"
        proj = dropout_layer(proj, use_noise, rrng, 0.5)

    return proj

# nhd -> number of hidden units -> dim(output)
# state_below -> (Max No. of words in batch, No. of Samples, 300) -> the Input of LSTM
# (tparams = tnewp, state_below = emb11, options, prefix = '1lstm1', mask = mymaks, nhd =50)
def lstm_layer2(tparams, state_below, options, prefix='lstm', mask=None, nhd=None):
    # nsteps: Max No. of words in batch (maxlen, timesteps)
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # Define the machenism of LSTM units
    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')].T)
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, nhd))
        f = tensor.nnet.sigmoid(_slice(preact, 1, nhd))
        o = tensor.nnet.sigmoid(_slice(preact, 2, nhd))
        c = tensor.tanh(_slice(preact, 3, nhd))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        # there are two output for each lstm unit:
        # hidden layer output (h) and cell memory output (c)
        return [h, c]

    # state_below -> word2vec embedding
    # Re new the state_below to -> Wx_t + b ???
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')].T) +
                   tparams[_p(prefix, 'b')].T)

    #print "hvals"
    dim_proj = nhd

    # 为LSTM单元做正向传播（nsteps[timesteps]）次，每次san都会输出一个(N, 50)的矩阵
    # 最后一次循环（nsteps[timestpes]）才得到所有的hidden layer的数值
    # Forward propogation for nsteps, where the last loop results are expected
    # nsteps = state_below.shape[0]　== Max No. of words in batch == [timesteps]
    # state_below -> (Max No. of words in batch[timesteps], No. of Samples[N], 300)
    # hvals: values of the hidden layer, shape = (nsteps[timestep], n_samples[N], dim_proj[dim(output)])
    # yvals: values of output layers, shape = (nsteps[timestep], n_samples[N], dim_proj[dim(output)])
    [hvals,yvals], updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)

    # Here, the shape of hvals would be (nsteps[timestep], n_samples[N], dim_proj[dim(output)])
    return hvals

# IN USE
def adadelta(lr, tparams, grads, emb11,mask11,emb21,mask21,y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, (0.95 * rg2 + 0.05* (g ** 2)))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([emb11,mask11,emb21,mask21,y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, (0.95 * ru2 + 0.05 * (ud ** 2)))
             for ru2, ud in zip(running_up2,updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

# NOT USED
def sgd(lr, tparams, grads, emb11,mask11,emb21,mask21,y, cost):

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function([emb11,mask11,emb21,mask21,y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')
    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

# NOT USED
def rmsprop(lr, tparams, grads, emb11,mask11,emb21,mask21,y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([emb11,mask11,emb21,mask21,y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update



class LSTM():
    def __init__(self, nam, W=0, maxlen=0, load=False, training=False):

        self.W = W
        # 创建2个LSTM单元（参数：WUb）放入词典中，并初始化参数
        # Generate 2 LSTM unit with Guassian innitialization
        # Type: Dictionary
        self.maxlen = maxlen
        newp = creatrnnx()
        self.model_name = nam
        # 让两个LSTM单元的参数WUb的初始相同
        # Make the weights(WUb) of both LSTM unit same
        for i in newp.keys():
            if i[0] == '1':
                newp['2' + i[1:]] = newp[i]

        # Create 5 tensors (symoblic) variables (y, mask11, mask21, emb11, emb21)
        # Here, config.floatX = 'float32'
        y = T.vector('y', dtype = config.floatX)
        mask11 = T.matrix('mask11', dtype = config.floatX)
        mask21 = T.matrix('mask21', dtype = config.floatX)
        emb11 = T.ftensor3('emb11')
        emb21 = T.ftensor3('emb21') # 3-D float-type tensor

        # Load the existed model (pre-trained weights) if needed
        if load == True:
            newp = pickle.load(open(nam,'rb'))

        # Convert 'newp' to shared-tensor-type dictionary 'tnewp'
        # Shared tenasor variable
        self.tnewp = init_tparams(newp)

        # Set tensor-type noise
        use_noise = theano.shared(numpy_floatX(0.))

        # Set tensor-type random number generator
        # rng -> random number generator
        trng = RandomStreams(1234)

        # ??? rrng?
        # create a 3-D random tensor for "dropout"?
        rate = 0.5
        rrng = trng.binomial(emb11.shape, p = 1 - rate, n = 1, dtype = emb11.dtype)
        # print "rrng:"
        # print "type of rrng:", type(rrng)
        # print rrng

        # 具体化LSTM模型的结构和参数（核心）proj代表着一个mini-batch输入以后的输出值
        # Implement the LSTM module;
        # Here 'False' -> NOT apply DROPOUT layers;
        # Since the input is in the format: (Max No. of words in batch, No. of Samples, 300)
        # Note: that the 1st term and 2nd term are exchanged!
        # 只需要getp()即scan循环以后的最后一次（timesteps）结果，之前记录LSTM输出的结果都抛弃
        # proj11[-1] -> (No. of samples[N], Hidden unit dimension[timesteps]) -> (N, 50)
        # proj11 takes the inputs as embedding matrix emb1 and gives the o/p of the LSTM_A
        proj11 = getpl2(emb11, '1lstm1', mask11, False, rrng, 50, self.tnewp)[-1]
        proj21 = getpl2(emb21, '2lstm1', mask21, False, rrng, 50, self.tnewp)[-1]

        # Define the cost function
        dif = (proj21 - proj11).norm(L = 1, axis = 1)
        s2 = T.exp(-dif)
        sim = T.clip(s2, 1e-7, 1.0-1e-7) # Similarity
        lr = tensor.scalar(name = 'lr') # learning rate
        ys = T.clip((y-1.0) / 4.0, 1e-7, 1.0-1e-7)
        cost = T.mean((sim - ys) ** 2)
        ns=emb11.shape[1]
        self.f2sim = theano.function([emb11, mask11, emb21, mask21], sim, allow_input_downcast = True)
        self.f_proj11 = theano.function([emb11, mask11], proj11, allow_input_downcast = True) # NOT used
        self.f_cost = theano.function([emb11, mask11, emb21, mask21, y], cost, allow_input_downcast = True) # NOT used

        # Prepare for the backpropogation and gradiant descend
        if training == True:

            # 计算cost对不同参数的导数，并且平均两个LSTM模型的参数
            # The gradi refers to gradients wrt. cost, and is a list containing gradients to be update weights
            # We average out the gradients by appending to another list grads[]
            # So, we average out the gradients : wrt LSTM_A and wrt LSTM_B
            # i.e, gradient= (grad(wrt(LSTM_A)+grad(wrt(LSTM_B))/2.0 to maintain the symmetricity between either LSTMs
            # wrt: (variable or list of variables) – term[s] for which we want gradients
            gradi = tensor.grad(cost, wrt = self.tnewp.values()) # T.grad -> differential
            grads = []
            l = len(gradi)
            for i in range(0, l/2):
                gravg = (gradi[i] + gradi[i + l / 2]) / (4.0)
            #print i,i+9
                grads.append(gravg)
            for i in range(0, len(self.tnewp.keys()) / 2):
                    grads.append(grads[i])

            # Here, the f_grad_shared and f_update are theano functions
            self.f_grad_shared, self.f_update = adadelta(lr, self.tnewp, grads, emb11, mask11, emb21, mask21, y, cost)


    def train_lstm(self, train, max_epochs, correct, test_correct, batchsize=32):
        print "Training"
        print "the length of the training data is ", len(train)

        # test = train

        print "Batchsize =", batchsize
       	print "max_epochs =", max_epochs
        lrate = 0.0001 # Learning rate, but Not USED ???
        freq = 0 # ???
        batchsize = 64
        dfreq = 21 #display frequency

        self.mse = [] # MSE of train1 + train2
        self.rank = []
        self.tops = {}

        self.mse_test = [] # MSE of test1
        self.mse_train = [] # MSE of train1
        self.rank_test = []
        self.tops_test = {}

        self.top_keys = [1, 5, 10]

        print "Before trianing, the error is:"
        # print self.chkterr2(train) # MSE check
        cst_all = self.chkterr2(train)[0]/16
        self.mse.append(cst_all)
        cst_test = self.chkterr2(test_correct)[0]/16
        self.mse_test.append(cst_test)
        cst_train = self.chkterr2(correct)[0]/16
        self.mse_train.append(cst_train)
        # 【注意】内存不足时使用chkterr2但是会慢，内存足够时使用 , self.get_mse(train)
        # 【注意】不要直接使用cst变量作为cost，因为这里的cst是最后一个batch的cost而已，不是全部的
        print "Training error:", cst_all #, "==", self.get_mse(train)
        print "Training_correct error", cst_train
        print "Testing_correct error:", cst_test


        # Saving (Initialization) the ranking and top1,5,10 information (Trianing data)
        rank_results_train, n_tops = self.evaluate2(correct, tops=self.top_keys) # Similairty check
        # print "[debug]", n_tops
        for top_key in self.top_keys:
            # print "[debug]", n_tops[top_key]
            self.tops[top_key] = []
            self.tops[top_key].append(n_tops[top_key])
            print "top-",top_key, "=", self.tops[top_key], ":", n_tops[top_key]
        print "Discription of evaluation (ranking) for training data:"
        print pd.Series(rank_results_train).describe()

        # Saving (Initialization) the ranking and top1,5,10 information (Testing data)
        rank_results_test, n_tops_test = self.evaluate2(test_correct, tops=self.top_keys) # Similairty check
        # print "[debug]", n_tops
        for top_key in self.top_keys:
            # print "[debug]", n_tops[top_key]
            self.tops_test[top_key] = []
            self.tops_test[top_key].append(n_tops_test[top_key])
            print "top-",top_key, "=", self.tops_test[top_key], ":", n_tops_test[top_key]
        print "Discription of evaluation (ranking) for testing data:"
        print pd.Series(rank_results_test).describe()

        # eidx -> index of epoch
        for eidx in xrange(0, max_epochs):
            sta = time.time()
            print ""
            print 'Epoch', eidx, '...'

            num = len(train) # length of training data

            #---------------------Shuffle the data------------------------------#
            # 为何不直接用shuffle函数？
            # generates a list with length of num from the population xrange(num)
            # Used for shuffling the training data each time for each epoches
            # [5,2,6,.11,...] length -> len(train)
            rnd = random.sample(xrange(num), num)

            # i would be (0,32,64,...)
            # Iterate all batches
            for i in range(0, num, batchsize):
                q = []
                x = i + batchsize
                if x > num:
                    x = num

                # Shuffle data
                # Iterate samples inside each batch
                # i -> start index of the batch
                # x -> end index of the batch
                for z in range(i, x):
                    # shuffling the training data to the list q
                    q.append(train[rnd[z]])
            #---------------------------------------------------------------------#

                """
                Mask for LSTM is prepared by sentence module
                x1 = np.array([["我"，"很"，"好"，"，"，"，"，"，"][...]...])
                len(x1) => 文档的总数
                mas1 = np.array([[1,1,1,0,0,0,0,0,0,0][...]...])
                """
                x1, mas1, x2, mas2, y2 = prepare_data(q, self.maxlen)

                ls = []
                ls2 = []
                freq += 1
                use_noise.set_value(1.)
                for j in range(0, len(x1)):
                    ls.append(embed(x1[j], 'en', W=self.W))
                    ls2.append(embed(x2[j], 'jp'))
                trconv = np.dstack(ls)
                trconv2 = np.dstack(ls2)
                emb2 = np.swapaxes(trconv2, 1, 2)
                emb1 = np.swapaxes(trconv, 1, 2)

                cst = self.f_grad_shared(emb2, mas2, emb1, mas1, y2)
                s = self.f_update(lrate) # Not USED ???

                if np.mod(freq, dfreq) == 0:
                    print 'Epoch ', eidx, 'Update ', freq, 'Cost ', cst
                # print 'Epoch ', eidx, 'Update ', freq, 'Cost ', cst

            # Evalution
            # print self.chkterr2(train) # MSE check
            cst_all = self.chkterr2(train)[0]/16
            self.mse.append(cst_all)
            cst_test = self.chkterr2(test_correct)[0]/16
            self.mse_test.append(cst_test)
            cst_train = self.chkterr2(correct)[0]/16
            self.mse_train.append(cst_train)
            # 【注意】内存不足时使用chkterr2但是会慢，内存足够时使用 , self.get_mse(train)
            # 【注意】不要直接使用cst变量作为cost，因为这里的cst是最后一个batch的cost而已，不是全部的
            # 错误用法： print "Training error:", cst, "=", self.chkterr2(train)[0]/16, "==", self.get_mse(train)
            print "Training error:", cst_all #, "==", self.get_mse(train)
            print "Training_correct error", cst_train
            print "Testing_correct error:", cst_test


            # Saving the ranking and top1,5,10 information
            rank_results_train, n_tops = self.evaluate2(correct, tops=self.top_keys) # Similairty check
            self.rank.append(rank_results_train)
            for top_key in self.top_keys:
                self.tops[top_key].append(n_tops[top_key])
                print "top-",top_key, "=", self.tops[top_key], ":", n_tops[top_key]
            print "Discription of evaluation (ranking) for training data:"
            print pd.Series(rank_results_train).describe()

            # Saving the ranking and top1,5,10 information
            rank_results_test, n_tops_test = self.evaluate2(test_correct, tops=self.top_keys) # Similairty check
            self.rank_test.append(rank_results_test)
            for top_key in self.top_keys:
                self.tops_test[top_key].append(n_tops_test[top_key])
                print "top-",top_key, "=", self.tops_test[top_key], ":", n_tops_test[top_key]
            print "Discription of evaluation (ranking) for testing data:"
            print pd.Series(rank_results_test).describe()

            # Saving the present weights:
            self.save_model(name=self.model_name+"_"+str(eidx)+".p")

            sto = time.time()
            self.time_saver = sto - sta
            print "epoch took:", self.time_saver

    # --- check the error 2 ---#
    # 【注意】这个函数之所以效率低下，要每256组数据为一个循环来做数据的预测 -> 为了防止内存不足！！
    def chkterr2(self, mydata):
        # count = []
        num = len(mydata)
        px = []
        yx = []
        use_noise.set_value(0.)
        for i in range(0, num, 256):
            q = []
            x=i + 256
            if x > num:
                x = num
            for j in range(i, x):
                q.append(mydata[j])
            x1,mas1,x2,mas2,y2 = prepare_data(q, self.maxlen)
            ls = []
            ls2 = []
            for j in range(0, len(q)):
                ls.append(embed(x1[j], 'en', W=self.W))
                ls2.append(embed(x2[j], 'jp'))
            trconv = np.dstack(ls)
            trconv2 = np.dstack(ls2)
            emb2 = np.swapaxes(trconv2, 1, 2)
            emb1 = np.swapaxes(trconv, 1, 2)
            pred = (self.f2sim(emb1, mas1, emb2, mas2)) * 4.0 + 1.0
            #dm1=np.ones(mas1.shape,dtype=np.float32)
            #dm2=np.ones(mas2.shape,dtype=np.float32)
            #corr=f_cost(emb1,mas1,emb2,mas2,y2)
            for z in range(0, len(q)):
                yx.append(y2[z])
                px.append(pred[z])
        #count.append(corr)
        px = np.array(px)
        yx = np.array(yx)
        #print "average error= "+str(np.mean(acc))
        return np.mean(np.square(px - yx)), meas.pearsonr(px, yx)[0],meas.spearmanr(yx, px)[0]

    def predict_similarity(self, sa, sb):
        q=[[sa, sb, 0]]
        x1, mas1, x2, mas2, y2 = prepare_data(q, self.maxlen)
        ls = []
        ls2 = []
        use_noise.set_value(0.)
        for j in range(0, len(x1)):
            ls.append(embed(x1[j], 'en', W=self.W))
            ls2.append(embed(x2[j], 'jp'))
        trconv = np.dstack(ls)
        trconv2 = np.dstack(ls2)
        emb2 = np.swapaxes(trconv2, 1, 2)
        emb1 = np.swapaxes(trconv, 1, 2)
        return self.f2sim(emb1, mas1, emb2, mas2)

    def save_model(self, type = 'pikcle', name=None):
        if name == None:
            name = self.model_name
    	print "Saving the model to", name
        self.new_params = unzip(self.tnewp)
        print "saving the model..."
        with open(name, 'wb') as handle:
            pickle.dump(self.new_params, handle)


    # Evaluate the each pairs of multilingual language
    # Give each pair a similairty ranking (for 1-999)
    def evaluate(self, data):
        x1, mas1, x2, mas2, y2 = prepare_data(data, self.maxlen)
        use_noise.set_value(0.)

        n_samples = len(data)

        ls = []   # Embedding results of xa
        ls2 = []  # Embedding results of xb
        for j in range(0, n_samples):
            ls.append(embed(x1[j], 'en', W=self.W))
            ls2.append(embed(x2[j], 'jp'))

        # print "ls: (should be the same ref_embed)", ls
        rank_results = []

        for i in range(0, n_samples):

            # NOTE: mas1 and mas2 are verticle matrix, not a normal one!
            # ref_ls refers to n_samples(999,EN) of duplicated ls[i]
            # So we can compare the ls[i](EN) with other sentences(999,JP)
            # to derive the ranking results for this given article ls[i](EN)
            # 用一个英语文章比较所有可能为pairs的日语文章（如999篇）求出ranking
            # ref_ls 就是一个重复了999（n_samples）次的文章ls[i]
            # 而 ls2 就是可能为paris的999篇日语的文章
            ref_ls = [ls[i]] * n_samples
            # print "ref_embed", ref_embed
            ref_mas1 = np.array([mas1[:,i],] * n_samples).T
            # print "ref_mas", ref_mas
            # print "mas1", mas1
            # return mas1, ref_mas
            trconv = np.dstack(ref_ls)
            trconv2 = np.dstack(ls2)
            emb2 = np.swapaxes(trconv2, 1, 2)
            emb1 = np.swapaxes(trconv, 1, 2)
            pred = self.f2sim(emb1, ref_mas1, emb2, mas2)

            rank = pd.Series(pred).rank(ascending = False)[i]
            rank_results.append(rank)
            print "the round", i, "rank:", rank

        return rank_results

    # project a list of article (cluster numbers) to 50 dim vectors
    def seq2vec(self, data):
        # list saving the projection results (50 dim):

        x1, mas1, x2, mas2, y2 = prepare_data(data, self.maxlen)
        # print "Finish preparing the data!"
        use_noise.set_value(0.)

        n_samples = len(data)

        ls = []   # Embedding results of xa
        ls2 = []  # Embedding results of xb
        for j in range(0, n_samples):
            ls.append(embed(x1[j], 'en', W=self.W))
            ls2.append(embed(x2[j], 'jp'))

        # print "Finished embedding,start projecting..."

        # start_time = time.time()
        # for i in range(0, n_samples):

        # print "conducting the", i, "projection"
        # loop_time = time.time()

        trconv = np.dstack(ls)
        trconv2 = np.dstack(ls2)

        emb1 = np.swapaxes(trconv, 1, 2)
        emb2 = np.swapaxes(trconv2, 1, 2)

        # list saving the projection results (50 dim):
        list_projection1 = self.f_proj11(emb1, mas1)
        list_projection2 = self.f_proj11(emb2, mas2)

        # After projection, compare the distance for possible pairs
        # ## SKIP



        return list_projection1, list_projection2

    # Example: tops = [1, 5, 10]
    def evaluate2(self, data, tops):
        projection1_train, projection2_train = self.seq2vec(data)
        # projection1_test, projection2_test = sls.seq2vec(test_1)

        # Calculate the rankings for this data set
        sim_results_train, rank_results_train = find_ranking(projection1_train, projection2_train)

        # Calculate the top1, top5 and top10 information
        n_tops = {}
        for top in tops:
            n_tops[top] = find_top(rank_results_train, top)

        return rank_results_train, n_tops


    def get_mse(self, data):
        # list saving the projection results (50 dim):

        x1, mas1, x2, mas2, y2 = prepare_data(data, self.maxlen)
        # print "Finish preparing the data!"
        use_noise.set_value(0.)

        n_samples = len(data)

        ls = []   # Embedding results of xa
        ls2 = []  # Embedding results of xb
        for j in range(0, n_samples):
            ls.append(embed(x1[j], 'en', W=self.W))
            ls2.append(embed(x2[j], 'jp'))

        # print "Finished embedding,start projecting..."

        # start_time = time.time()
        # for i in range(0, n_samples):

        # print "conducting the", i, "projection"
        # loop_time = time.time()

        trconv = np.dstack(ls)
        trconv2 = np.dstack(ls2)

        emb1 = np.swapaxes(trconv, 1, 2)
        emb2 = np.swapaxes(trconv2, 1, 2)

        # list saving the projection results (50 dim):

        # list_projection1 = self.f_proj11(emb1, mas1)
        # list_projection2 = self.f_proj11(emb2, mas2)
        c = self.f_cost(emb1, mas1, emb2, mas2, y2)

        # After projection, compare the distance for possible pairs
        # ## SKIP



        return c

