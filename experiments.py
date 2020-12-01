# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:51:32 2018

@author: ifbd
"""

import numpy as np
from pandas import DataFrame as df
from utils import acc,recall,precision,recall_n_top
import data
import matplotlib.pyplot as plt
import os


def integrate_results(record_file,model_list,data_list,topic_list,test_folds,thre_rating=0.5):
    '''
    calculate the average value of each test_fold
    '''
    record=dict()
    record['model']=[]
    record['data']=[]
    record['topic']=[]
    record['recall']=[]
    record['precision']=[]
    record['accuracy']=[]

    for model_name in model_list:
        for data_name in data_list:
            for n_topic in topic_list:
                recall_avg=0
                precision_avg=0
                accuracy_avg=0
                n=0
                for test_fold in range(1,test_folds+1):
                    if data_name=="citeulike-a":
                        num_user=5551
                        num_item=16980
                        test_file='data\\cf-a\\cf-a-test-'+str(test_fold)+'-users.dat'
                    elif data_name=="citeulike-t":
                        num_user=7947
                        num_item=25975
                        test_file='data\\cf-t\\cf-t-test-'+str(test_fold)+'-users.dat'
                
                    test_R,test_user_set,test_item_set,num_test_ratings,test_C=data.read_user(test_file,num_user,num_item)
                    test_mask_R=test_R
                    result_path='results/'+model_name+'/'+data_name+'/variable/'
                    estimated_file=result_path+str(n_topic)+'-'+str(test_fold)+'-Estimated_R.txt'
                    print(estimated_file)
                    if os.path.exists(estimated_file):
                        
                        estimated_R=np.loadtxt(estimated_file)
                        
                        Recall=recall(test_R,test_mask_R,estimated_R,num_test_ratings,thre_rating)

                        Precision=precision(test_R,test_mask_R,estimated_R,num_test_ratings,thre_rating)
                        Accuracy=acc(test_R,test_mask_R,estimated_R,num_test_ratings,thre_rating)
                    
                        recall_avg+=Recall
                        precision_avg+=Precision
                        accuracy_avg+=Accuracy
                        n+=1
                        print(n)
                if n>0:    
                    record['model'].append(model_name)
                    record['data'].append(data_name)
                    record['topic'].append(n_topic)
                    record['recall'].append(recall_avg/n)
                    record['precision'].append(precision_avg/n)
                    record['accuracy'].append(accuracy_avg/n)
    record_df=df(record)
    record_df.to_csv(record_file,index=False)



def two_way_decision_cost(test_R,test_mask_R,Estimated_R,num_test_ratings,cost_dict,thre_rating,sparse):
    is_test_user=test_R.sum(axis=1)
    is_test_user[is_test_user>0]=1
    is_test_user=is_test_user.reshape((len(is_test_user),1))

    test_user=np.argwhere(is_test_user!= 0).flatten().tolist()
    n_user,n_item=Estimated_R.shape
    
    tmp_pre_R=np.sign(Estimated_R-thre_rating)
    tmp_test_R = np.sign(test_R - thre_rating)
    
    pre_rec_ratings=np.sum(np.multiply(tmp_pre_R==1,is_test_user))
    test_rec_ratings=np.sum(np.multiply(tmp_test_R==1,test_mask_R))
    true_pre_rec_ratings= np.sum(np.multiply(((tmp_pre_R == tmp_test_R) & (tmp_pre_R ==1)), test_mask_R))    
     
    mis_cost=(sparse*pre_rec_ratings-true_pre_rec_ratings)*cost_dict['RD']+(test_rec_ratings-true_pre_rec_ratings)*cost_dict['NL']
     
    mis_cost=mis_cost/len(test_user)
    
    return mis_cost

                  
def three_way_decision_cost(test_R,test_mask_R,Estimated_R,num_test_ratings,cost_dict,thre_rating,sparse,alpha,beta):
    
    tmp_pre_R=np.zeros(Estimated_R.shape)
    tmp_pre_R[Estimated_R>=alpha]=1
    tmp_pre_R[Estimated_R<=beta]=-1
    
    tmp_test_R = np.sign(test_R - thre_rating)
    
    is_test_user=test_R.sum(axis=1)
    is_test_user[is_test_user>0]=1
    is_test_user=is_test_user.reshape((len(is_test_user),1))
    
    n_positive=np.sum(np.muiliply(tmp_pre_R==1,is_test_user))
    n_boundary=np.sum(np.muiliply(tmp_pre_R==0,is_test_user))
    n_negative=np.sum(np.muiliply(tmp_pre_R==-1,is_test_user))

    test_rec_ratings=np.sum(np.multiply(tmp_test_R==1,test_mask_R))
    true_pre_rec_ratings= np.sum(np.multiply(((tmp_pre_R == tmp_test_R) & (tmp_pre_R ==1)), test_mask_R))
    

    mis_cost=(sparse*n_positive-true_pre_rec_ratings)*cost_dict['RD']+(test_rec_ratings-true_pre_rec_ratings)*cost_dict['NL']
    teach_cost=n_boundary*cost_dict['BD']
    
    mis_cost=mis_cost
    teach_cost=teach_cost

    return n_positive,n_boundary,n_negative,mis_cost,teach_cost
    
    
def teaching_cost(n_topic,first_hidden_nueron,num_user,num_item,num_voca,lamda_BD,theta):
    '''
    theta 学习成本前面的系数
    '''
    teaching_cost_fix=theta*num_item*first_hidden_nueron*num_voca
    teaching_cost=teaching_cost_fix+theta*(pow(num_item,2)+pow(num_user,2))*pow(n_topic,2)
    return teaching_cost
    
    
if __name__=="__main__":
    model_list=['CTR','CDL']
    #data_list=["citeulike-a","citeulike-t"]
    data_list=["citeulike-t"]
    max_topic=100
    #topic_list=[i for i in range (5,max_topic+1,5)]
    topic_list=[10]
    test_folds=5
    record_file='results/intergrate_results.csv'
    integrate_results(record_file,model_list,data_list,topic_list,test_folds,thre_rating=0.5)
    

'''
theta=1e-10
theta=1e-11
plt.figure(1)
ax = plt.subplot(111)
X=np.linspace(5,200,1000)
Y=np.array([teaching_cost(x,first_hidden_nueron,num_user,num_item,num_voca,lamda_BD,theta) for x in X])
ax.plot(X, Y)


model_name='CDL'
data_name='citeulike-a'
test_fold=1
thre_rating=0.5
cost_dict={'RL':0,'RD':100,'BL':1,'BD':1,'NL':10,'ND':0}

if data_name=="citeulike-a":
    num_user=5551
    num_item=16980
    num_voca=8000  
    num_ratings=204987
    train_file='data\\cf-a\\cf-a-train-'+str(test_fold)+'-users.dat'
    test_file='data\\cf-a\\cf-a-test-'+str(test_fold)+'-users.dat'
    item_info='data\\cf-a\\cf-a-mult.dat'
elif data_name=="citeulike-t":
    num_user=7947
    num_item=25975
    num_voca=20000 
    num_ratings=134860
    train_file='data\\cf-t\\cf-t-train-'+str(test_fold)+'-users.dat'
    test_file='data\\cf-t\\cf-t-test-'+str(test_fold)+'-users.dat'
    item_info='data\\cf-t\\cf-t-mult.dat'

sparse=num_ratings/(num_user*num_item)
test_R,test_user_set,test_item_set,num_test_ratings,test_C=data.read_user(test_file,num_user,num_item)
test_mask_R=test_R
result_path='results/'+model_name+'/'+data_name+'/variable/'
estimated_file=result_path+'5-Estimated_R.txt'
estimated_R=np.loadtxt(estimated_file)
alpha=0.8
beta=0.2

Recall=recall(test_R,test_mask_R,estimated_R,num_test_ratings,thre_rating)
Precision=precision(test_R,test_mask_R,estimated_R,num_test_ratings,thre_rating)
Accuracy=acc(test_R,test_mask_R,estimated_R,num_test_ratings,thre_rating)

'''
