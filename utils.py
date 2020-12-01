# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:41:06 2018

@author: ifbd
"""

import numpy as np
import os
from numpy import inf
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import batch_norm
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


## variation in the aspect of train epoch
def make_records(result_path,test_acc_list,test_rmse_list,test_mae_list,test_avg_loglike_list,test_recall_list,
                 test_precision_list,test_map_list,test_ndcg_one_list,test_ndcg_two_list,test_converage_list,test_diversity_list,
                 current_time,args,model_name,data_name,train_ratio,hidden_neuron,random_seed,optimizer_method,lr):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    time.sleep(5)
    overview = './results/' +str(hidden_neuron)+"-"+ 'overview.txt'
    basic_info = result_path +str(hidden_neuron)+"-"+"basic_info.txt"
    test_record = result_path +str(hidden_neuron)+"-"+ "test_record.txt"

    with open(test_record, 'w') as g:

        g.write(str("ACC:"))
        g.write('\t')
        for itr in range(len(test_acc_list)):
            g.write(str(test_acc_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("RMSE:"))
        g.write('\t')
        for itr in range(len(test_rmse_list)):
            g.write(str(test_rmse_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("MAE:"))
        g.write('\t')
        for itr in range(len(test_mae_list)):
            g.write(str(test_mae_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("AVG Likelihood:"))
        g.write('\t')
        for itr in range(len(test_avg_loglike_list)):
            g.write(str(test_avg_loglike_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("Recall:"))
        g.write('\t')
        for itr in range(len(test_recall_list)):
            g.write(str(test_recall_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("Precision:"))
        g.write('\t')
        for itr in range(len(test_precision_list)):
            g.write(str(test_precision_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("MAP:"))
        g.write('\t')
        for itr in range(len(test_map_list)):
            g.write(str(test_map_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("NDCG_one:"))
        g.write('\t')
        for itr in range(len(test_ndcg_one_list)):
            g.write(str(test_ndcg_one_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("NDCG_two:"))
        g.write('\t')
        for itr in range(len(test_ndcg_two_list)):
            g.write(str(test_ndcg_two_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("Converage:"))
        g.write('\t')
        for itr in range(len(test_converage_list)):
            g.write(str(test_converage_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("Diversity:"))
        g.write('\t')
        for itr in range(len(test_diversity_list)):
            g.write(str(test_diversity_list[itr]))
            g.write('\t')
        g.write('\n')
        
    with open(basic_info, 'w') as h:
        h.write(str(args))
        

    with open(overview, 'a') as f:
        f.write(str(data_name))
        f.write('\t')
        f.write(str(model_name))
        f.write('\t')
        f.write(str(train_ratio))
        f.write('\t')
        f.write(str(current_time))
        f.write('\t')
        f.write(str(test_rmse_list[-1]))
        f.write('\t')
        f.write(str(test_mae_list[-1]))
        f.write('\t')
        f.write(str(test_acc_list[-1]))
        f.write('\t')
        f.write(str(test_avg_loglike_list[-1]))
        f.write('\t')
        f.write(str(hidden_neuron))
        f.write('\t')
        f.write(str(args.corruption_level))
        f.write('\t')

        f.write(str(args.lambda_u))
        f.write('\t')
        f.write(str(args.lambda_w))
        f.write('\t')
        f.write(str(args.lambda_n))
        f.write('\t')
        f.write(str(args.lambda_v))
        f.write('\t')
        f.write(str(args.f_act))
        f.write('\t')
        f.write(str(args.g_act))
        f.write('\n')

    Test = plt.plot(test_acc_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig(result_path+str(hidden_neuron)+"_" + "ACC.png")
    plt.clf()

    Test = plt.plot(test_rmse_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(result_path+str(hidden_neuron)+"_"+ "RMSE.png")
    plt.clf()

    Test = plt.plot(test_mae_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(result_path+str(hidden_neuron)+"_" + "MAE.png")
    plt.clf()

    Test = plt.plot(test_avg_loglike_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test AVG likelihood')
    plt.legend()
    plt.savefig(result_path+str(hidden_neuron)+"_" + "AVG.png")
    plt.clf()
    
    Test = plt.plot(test_recall_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test Recall')
    plt.legend()
    plt.savefig(result_path +str(hidden_neuron)+"_"+ "Recall.png")
    plt.clf()
    
    Test = plt.plot(test_precision_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test Precision')
    plt.legend()
    plt.savefig(result_path +str(hidden_neuron)+"_"+"Precision.png")
    plt.clf()
    
    Test = plt.plot(test_map_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test MAP')
    plt.legend()
    plt.savefig(result_path +str(hidden_neuron)+"_"+"MAP.png")
    plt.clf()
    
    Test = plt.plot(test_ndcg_one_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test NDCG_one')
    plt.legend()
    plt.savefig(result_path +str(hidden_neuron)+"_"+"NDCG_one.png")
    plt.clf()
    
    Test = plt.plot(test_ndcg_two_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test NDCG_two')
    plt.legend()
    plt.savefig(result_path +str(hidden_neuron)+"_"+"NDCG_two.png")
    plt.clf()
    
    Test = plt.plot(test_converage_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test Converage')
    plt.legend()
    plt.savefig(result_path +str(hidden_neuron)+"_"+"Converage.png")
    plt.clf()
    
    Test = plt.plot(test_diversity_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test Diversity')
    plt.legend()
    plt.savefig(result_path +str(hidden_neuron)+"_"+"Diversity.png")
    plt.clf()



## variation in the aspect of n_top
def make_records_n_top(result_path,fin_acc_list,fin_rmse_list,fin_mae_list,fin_avg_loglike_list,fin_recall_list,
                 fin_precision_list,fin_map_list,fin_ndcg_one_list,fin_ndcg_two_list,fin_converage_list,fin_diversity_list,
                 model_name,data_name,hidden_factor,test_fold):
    result_path=result_path+'measure/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    fin_record = result_path +str(hidden_factor)+"-"+str(test_fold)+"-"+ "fin_record.txt"
    time.sleep(5)
    with open(fin_record, 'w') as g:

        g.write(str("ACC:"))
        g.write('\t')
        for itr in range(len(fin_acc_list)):
            g.write(str(fin_acc_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("RMSE:"))
        g.write('\t')
        for itr in range(len(fin_rmse_list)):
            g.write(str(fin_rmse_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("MAE:"))
        g.write('\t')
        for itr in range(len(fin_mae_list)):
            g.write(str(fin_mae_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("AVG Likelihood:"))
        g.write('\t')
        for itr in range(len(fin_avg_loglike_list)):
            g.write(str(fin_avg_loglike_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("Recall:"))
        g.write('\t')
        for itr in range(len(fin_recall_list)):
            g.write(str(fin_recall_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("Precision:"))
        g.write('\t')
        for itr in range(len(fin_precision_list)):
            g.write(str(fin_precision_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("MAP:"))
        g.write('\t')
        for itr in range(len(fin_map_list)):
            g.write(str(fin_map_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("NDCG_one:"))
        g.write('\t')
        for itr in range(len(fin_ndcg_one_list)):
            g.write(str(fin_ndcg_one_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("NDCG_two:"))
        g.write('\t')
        for itr in range(len(fin_ndcg_two_list)):
            g.write(str(fin_ndcg_two_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("Converage:"))
        g.write('\t')
        for itr in range(len(fin_converage_list)):
            g.write(str(fin_converage_list[itr]))
            g.write('\t')
        g.write('\n')
        
        g.write(str("Diversity:"))
        g.write('\t')
        for itr in range(len(fin_diversity_list)):
            g.write(str(fin_diversity_list[itr]))
            g.write('\t')
        g.write('\n')
        

def variable_save(result_path,model_name,train_var_list1,train_var_list2,Estimated_R,test_v_ud,mask_test_v_ud,hidden_neuron,test_fold):
    result_path=result_path+'variable/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    time.sleep(5)
    '''
    for var in train_var_list1:
        var_value = var.eval()
        var_name = ((var.name).split('/'))[1]
        var_name = (var_name.split(':'))[0]
        np.savetxt(result_path +str(hidden_neuron)+"-"+str(test_fold)+"-"+var_name , var_value)
   
    for var in train_var_list2:
        if model_name == "DIPEN_with_VAE":
            var_value = var.eval()
            var_name = (var.name.split(':'))[0]
            print (var_name)
            var_name = var_name.replace("/","_")
            #var_name = ((var.name).split('/'))[2]
            #var_name = (var_name.split(':'))[0]
            print (var.name)
            print (var_name)
            print ("================================")
            np.savetxt(result_path +str(hidden_neuron)+str(test_fold)+"-"+"-"+var_name, var_value)
        else:
            var_value = var.eval()
            var_name = ((var.name).split('/'))[1]
            var_name = (var_name.split(':'))[0]
            np.savetxt(result_path +str(hidden_neuron)+str(test_fold)+"-"+"-"+var_name , var_value)
    '''
    np.savetxt(result_path+str(hidden_neuron)+"-"+str(test_fold)+"-"+"Estimated_R.txt",Estimated_R)


def SDAE_calculate(model_name,X_c, layer_structure, W, b, batch_normalization, f_act,g_act, model_keep_prob,V_u=None):
    hidden_value = X_c
    for itr1 in range(len(layer_structure) - 1):
        ''' Encoder '''
        if itr1 <= int(len(layer_structure) / 2) - 1:
            if (itr1 == 0) and (model_name == "CDAE"):
                ''' V_u '''
                before_activation = tf.add(tf.add(tf.matmul(hidden_value, W[itr1]),V_u), b[itr1])
            else:
                before_activation = tf.add(tf.matmul(hidden_value, W[itr1]), b[itr1])
            if batch_normalization == "True":
                before_activation = batch_norm(before_activation)
            hidden_value = f_act(before_activation)
            ''' Decoder '''
        elif itr1 > int(len(layer_structure) / 2) - 1:
            before_activation = tf.add(tf.matmul(hidden_value, W[itr1]), b[itr1])
            if batch_normalization == "True":
                before_activation = batch_norm(before_activation)
            hidden_value = g_act(before_activation)
        if itr1 < len(layer_structure) - 2: # add dropout except final layer
            hidden_value = tf.nn.dropout(hidden_value, model_keep_prob)
        if itr1 == int(len(layer_structure) / 2) - 1:
            Encoded_X = hidden_value

    sdae_output = hidden_value

    return Encoded_X, sdae_output

def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

def softmax(w, t = 1.0):
    npa = np.array
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist




