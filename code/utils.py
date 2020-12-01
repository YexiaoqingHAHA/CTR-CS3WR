# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:41:06 2018

@author: ifbd
"""

import numpy as np
import os
from numpy import inf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time



def rmse(test_R,test_mask_R,Estimated_R,num_test_ratings):
    pre_numerator = np.multiply((test_R - Estimated_R), test_mask_R)
    numerator = np.sum(np.square(pre_numerator))
    RMSE = np.sqrt(numerator / float(num_test_ratings))
    return RMSE

def mae(test_R,test_mask_R,Estimated_R,num_test_ratings):
    pre_numeartor = np.multiply((test_R - Estimated_R), test_mask_R)
    numerator = np.sum(np.abs(pre_numeartor))
    MAE = numerator / float(num_test_ratings)
    return MAE

def acc(test_R,test_mask_R,Estimated_R,num_test_ratings,thre_rating):
    pre_numeartor1 = np.sign(Estimated_R - thre_rating)
    tmp_test_R = np.sign(test_R - thre_rating)

    pre_numerator2 = np.multiply((pre_numeartor1 == tmp_test_R), test_mask_R)
    numerator = np.sum(pre_numerator2)
    ACC = numerator / float(num_test_ratings)
    return ACC

def avg_loglikelihood(test_R,test_mask_R,Estimated_R,num_test_ratings):
    a = np.log(Estimated_R)
    b = np.log(1 - Estimated_R)
    a[a == -inf] = 0
    b[b == -inf] = 0

    tmp_r = test_R
    tmp_r = a * (tmp_r > 0) + b * (tmp_r == 0)
    tmp_r = np.multiply(tmp_r, test_mask_R)
    numerator = np.sum(tmp_r)
    AVG_loglikelihood = numerator / float(num_test_ratings)
    return AVG_loglikelihood

def recall(test_R,test_mask_R,Estimated_R,num_test_ratings,thre_rating):

    tmp_pre_R = np.sign(Estimated_R - thre_rating)
    tmp_test_R = np.sign(test_R - thre_rating)
    true_pre_rec_ratings= np.sum(np.multiply(((tmp_pre_R == tmp_test_R) & (tmp_pre_R ==1)), test_mask_R))
    true_rec_ratings=np.sum(np.multiply(tmp_test_R==1,test_mask_R))
    recall=true_pre_rec_ratings/true_rec_ratings
    return recall

def precision(test_R,test_mask_R,Estimated_R,num_test_ratings,thre_rating):
    is_test_user=test_R.sum(axis=1)
    is_test_user[is_test_user>0]=1
    is_test_user=is_test_user.reshape((len(is_test_user),1))
    tmp_pre_R = np.sign(Estimated_R - thre_rating)
    tmp_test_R = np.sign(test_R - thre_rating)
    true_pre_rec_ratings= np.sum(np.multiply(((tmp_pre_R == tmp_test_R) & (tmp_pre_R ==1)), test_mask_R))
    pre_rec_ratings=np.sum(np.multiply(tmp_pre_R==1,is_test_user))
    if pre_rec_ratings==0:
        precision=0
    precision=true_pre_rec_ratings/pre_rec_ratings
    return precision  

def recall_n_top(test_R,test_mask_R,Estimated_R,num_test_ratings,thre_rating,top_n):

    top_R=np.array(np.argsort(-Estimated_R)[:,:top_n])
    num_user,num_item=test_R.shape
    #transform the recommedation list to the array(num_user,num_item)
    rec_num=np.arange(num_user).repeat(top_n).reshape((num_user,top_n))
    rec_array=np.zeros((num_user,num_item))
    rec_array[rec_num,top_R]=1
    tmp_test_R = np.sign(test_R - thre_rating)
    pre_numerator = np.multiply((rec_array == tmp_test_R), test_mask_R)
    numerator=np.sum(pre_numerator)
    denominator=np.sum(np.multiply((tmp_test_R==1),test_mask_R))
    recall=numerator/denominator
    return recall

    
    pre_numerator = np.multiply((rec_array == tmp_test_R), test_mask_R)
    numerator=np.sum(pre_numerator)
    denominator=np.sum(np.multiply((tmp_test_R==1),test_mask_R))
    recall=numerator/denominator
    return recall
    
def precision_n_top(test_R,test_mask_R,Estimated_R,num_test_ratings,thre_rating,top_n):
    is_test_user=test_R.sum(axis=1)
    test_user=np.argwhere(is_test_user!= 0).flatten().tolist()
    top_R=np.array(np.argsort(-Estimated_R)[:,:top_n])
    num_user,num_item=test_R.shape
    rec_num=np.arange(num_user).repeat(top_n).reshape((num_user,top_n))
    rec_array=np.zeros((num_user,num_item))
    rec_array[rec_num,top_R]=1
    tmp_test_R=np.sign(test_R-thre_rating)
    pre_numerator=np.multiply((rec_array==tmp_test_R),test_mask_R)
    numerator=np.sum(pre_numerator)
    precision=numerator/(top_n*len(test_user))
    return precision


###基于顺序的recall    
def Map(test_R,test_mask_R,Estimated_R,num_test_ratings,thre_rating,top_n):
    is_test_user=test_R.sum(axis=1)
    test_user=np.argwhere(is_test_user!= 0).flatten().tolist()
    top_R=np.array(np.argsort(-Estimated_R)[:,:top_n])
    num_user,num_item=top_R.shape
    tmp_test_R=np.sign(test_R-thre_rating)
    sum_AP=0
    for u in test_user:
        hits=0
        sum_precs=0
        for i in range(len(top_R[u])):
            if tmp_test_R[u,i]==1:
                hits+=1
                sum_precs+=hits/(i+1.0)
        if hits>0:
            sum_AP+=hits/sum_precs
    if sum_AP>0:
        MAP=sum_AP/float(len(test_user))
    else:
        MAP=0
    return MAP

def ndcg_one(test_R,test_mask_R,Estimated_R,num_test_ratings,top_n):
    is_test_user=test_R.sum(axis=1)
    test_user=np.argwhere(is_test_user!= 0).flatten().tolist()
    num_user,num_item=test_R.shape
    top_R_test=np.array(np.argsort(np.multiply(Estimated_R,test_mask_R))[:,:top_n])
    rec_num=np.arange(num_user).repeat(top_n).reshape((num_user,top_n))
    pred_real=test_R[rec_num,top_R_test]
    sum_ndcg=0
    for u in test_user:
        dcg=0
        pred_real_u=pred_real[u]
        for (index,rel_rating) in enumerate(pred_real_u):
            dcg+=(rel_rating*np.reciprocal(np.log2(index+2)))
        idcg=0
        pred_real_u.sort()
        real_rank=pred_real_u[::-1]
        for (index,rel_rating) in enumerate(real_rank):
            idcg+=(rel_rating* np.reciprocal(np.log2(index+2)))
        sum_ndcg+=(dcg/idcg)
    ndcg=sum_ndcg/len(test_user)
    return ndcg    
            
def ndcg_two(test_R,test_mask_R,Estimated_R,num_test_ratings,top_n):
    is_test_user=test_R.sum(axis=1)
    test_user=np.argwhere(is_test_user!= 0).flatten().tolist()
    num_user,num_item=test_R.shape
    top_R_test=np.array(np.argsort(np.multiply(Estimated_R,test_mask_R))[:,:top_n])
    rec_num=np.arange(num_user).repeat(top_n).reshape((num_user,top_n))
    pred_real=test_R[rec_num,top_R_test]
    sum_ndcg=0
    for u in test_user:
        dcg=0
        pred_real_u=pred_real[u]
        for (index,rel_rating) in enumerate(pred_real_u):
            dcg+=((np.power(2,rel_rating)+1)*np.reciprocal(np.log2(index+2)))
        idcg=0
        pred_real_u.sort()
        real_rank=pred_real_u[::-1]
        for (index,rel_rating) in enumerate(real_rank):
            idcg+=((np.power(2,rel_rating)+1)* np.reciprocal(np.log2(index+2)))
        sum_ndcg+=(dcg/idcg)
    ndcg=sum_ndcg/len(num_user)
    return ndcg
    
    
    
def converage(test_R,test_mask_R,Estimated_R,num_test_ratings,top_n):
    top_R=np.array(np.argsort(-Estimated_R)[:,:top_n])
    num_user,num_item=test_R.shape
    rec_num=np.arange(num_user).repeat(top_n).reshape((num_user,top_n))
    rec_array=np.zeros((num_user,num_item))
    rec_array[rec_num,top_R]=1
    rec_item=np.sum(rec_array,axis=0)
    rec_item[rec_item>=1]=1
    converage=np.sum(rec_item)/num_item
    return converage

def diversity(test_R,test_mask_R,Estimated_R,num_test_ratings,top_n,item_data_dw):
    is_test_user=test_R.sum(axis=1)
    test_user=np.argwhere(is_test_user!= 0).flatten().tolist()
    item_similarity=np.corrcoef(item_data_dw)
    num_user,num_item=test_R.shape
    top_rec=np.array(np.argsort(-Estimated_R)[:,:top_n])
    diversity=0
    for u in test_user:
        rec=top_rec[u]
        rec_1=rec[0]
        similarity_all=0
        for rec_other in rec[1:]:
            similarity_all+=item_similarity[rec_1,rec_other]
        diversity+=(1-2*similarity_all/(top_n*(top_n-1)))
    diversity=diversity/len(test_user)
    return diversity
            
    
    
    

def evaluation(test_R,test_mask_R,Estimated_R,num_test_ratings,item_data_dw,top_n=30,thre_rating=0.5):
    
    RMSE=0
    MAE=0
    ACC=0
    AVG_loglikelihood=0
    Recall=0
    Precision=0
    MAP=0
    NDCG_one=0
    NDCG_two=0
    Converage=0
    Diversity=0
    
    #RMSE=rmse(test_R,test_mask_R,Estimated_R,num_test_ratings)
    #MAE=mae(test_R,test_mask_R,Estimated_R,num_test_ratings)
    ACC=acc(test_R,test_mask_R,Estimated_R,num_test_ratings,thre_rating)
    #AVG_loglikelihood=avg_loglikelihood(test_R,test_mask_R,Estimated_R,num_test_ratings)
    Recall=recall_n_top(test_R,test_mask_R,Estimated_R,num_test_ratings,thre_rating,top_n)
    Precision=precision_n_top(test_R,test_mask_R,Estimated_R,num_test_ratings,thre_rating,top_n)
    MAP=Map(test_R,test_mask_R,Estimated_R,num_test_ratings,thre_rating,top_n)
    #NDCG_one=ndcg_one(test_R,test_mask_R,Estimated_R,num_test_ratings,top_n)
    #NDCG_two=ndcg_two(test_R,test_mask_R,Estimated_R,num_test_ratings,top_n)
    Converage=converage(test_R,test_mask_R,Estimated_R,num_test_ratings,top_n)
    #Diversity=diversity(test_R,test_mask_R,Estimated_R,num_test_ratings,top_n,item_data_dw)


    return RMSE,MAE,ACC,AVG_loglikelihood,Recall,Precision,MAP,NDCG_one,NDCG_two,Converage,Diversity






