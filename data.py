import numpy as np
import os
from pandas import DataFrame as df
#from mult import read_mult


def downloadData():
    data_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/cdl'
    for filename in ('mult.dat', 'cf-train-1-users.dat', 'cf-test-1-users.dat', 'raw-data.csv'):
        if not os.path.exists(filename):
            os.system("wget %s/%s" % (data_url, filename))





def read_dummy_user():
    R = np.mat(np.random.rand(100,100))
    R[R<0.9] = 0
    R[R>0.8] = 1
    return R

def read_mult(file_name,num_voca):
    fp=open(file_name)
    lines=fp.readlines()
    X=np.zeros((len(lines),num_voca))
    for i,line in enumerate(lines):
        strs=line.strip().split(' ')[1:]
        for strr in strs:
            segs=strr.split(':')
            X[i,int(segs[0])]=float(segs[1])
    #arr_max=np.amax(X,axis=1)
    #X=(X.T/arr_max).T
    return X

def read_mult_CTR(file_name,num_voca):
    fp=open(file_name)
    lines=fp.readlines()
    word_ids=[]
    word_cnt=[]
    for i,line in enumerate(lines):
        strs=line.strip().split(' ')[1:]
        word_ids_one=[]
        word_cnt_one=[]
        for strr in strs:
            segs=strr.split(':')
            word_ids_one.append(int(segs[0]))
            word_cnt_one.append(float(segs[1]))
        word_ids.append(word_ids_one)
        word_cnt.append(word_cnt_one)
    return word_ids,word_cnt
    

def get_mult(file_name,num_voca):
    X=read_mult(file_name,num_voca)
    return X

def get_dummy_mult():
    X=np.random.rand(100,100)
    X[X<0.9]=0
    return X

def read_user(rating_file,num_u,num_v,a=1,b=0.01):
    fp=open(rating_file)
    lines=fp.readlines()
    # np.mat转化为举证形式
    num_rating=0
    R=np.zeros([num_u,num_v])
    C=np.zeros([num_u,num_v])
    C=C+b
    user_set=set()
    item_set=set()
    for i,line in enumerate(lines):
        segs=line.strip().split(' ')[1:]
        for seg in segs:
            R[i,int(seg)]=1
            C[i,int(seg)]=a
            item_set.add(int(seg))
            num_rating+=1
        user_set.add(i)
        #print(i)
    return R,user_set,item_set,num_rating,C

def read_user_cmf(rating_file,num_u,num_v,a=1,b=0.01):
    fp=open(rating_file)
    lines=fp.readlines()
    # np.mat转化为举证形式
    num_rating=0
    R=np.ones([num_u,num_v])
    C=np.ones([num_u,num_v])
    C=C*b
    R=R*b
    user_set=set()
    item_set=set()
    for i,line in enumerate(lines):
        segs=line.strip().split(' ')[1:]
        for seg in segs:
            R[i,int(seg)]=1
            C[i,int(seg)]=a
            item_set.add(int(seg))
            num_rating+=1
        user_set.add(i)
        #print(i)
    return R,user_set,item_set,num_rating,C

def read_rating(rating_file):
    rating_dict=dict()
    rating_dict['UserId']=[]
    rating_dict['ItemId']=[]
    rating_dict['Rating']=[]

    fp=open(rating_file)
    lines=fp.readlines()
    for i,line in enumerate(lines):
        segs=line.strip().split(' ')[1:]
        for seg in segs:
            rating_dict['UserId'].append(i)
            rating_dict['ItemId'].append(int(seg))
            rating_dict['Rating'].append(1)
    rating_df=df(rating_dict)
    return rating_df
         

