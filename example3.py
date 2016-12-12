from lstm import *
training=True # Set to false to load weights
Syn_aug=True # it False faster but does slightly worse on Test dataset

sls=LSTM("new.p",load=False,training=True)

train=pickle.load(open("stsallrmf.p","rb"))#[:-8]
if training==True:
    print "Pre-training"
    sls.train_lstm(train,3) #66
    print "Pre-training done"
    train=pickle.load(open("semtrain.p",'rb'))
    if Syn_aug==True:
        train=expand(train)
        sls.train_lstm(train,100) # 340
    else:
        sls.train_lstm(train,100) # 330

test=pickle.load(open("semtest.p",'rb'))
print sls.chkterr2(test)
#Example
sa="A truly wise man"
sb="He is smart"
print sls.predict_similarity(sa,sb)*4.0+1.0