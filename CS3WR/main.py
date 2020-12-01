# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:53:43 2018

@author: ADMIN
"""

import numpy as np
import pandas as pd
from pandas import DataFrame as df
import data_process
from collections import defaultdict
from utils import recall_n_top, precision_n_top, recall, precision
from multiprocessing import Pool
import os
import time

def two_way_decision_cost_rating(test_R, Estimated_R, cost_dict, thre_rating,a,b,gamma):
    is_test_user = test_R.sum(axis=1)
    is_test_user[is_test_user > 0] = 1
    is_test_user = is_test_user.reshape((len(is_test_user), 1)).flatten()

    test_user = np.argwhere(is_test_user != 0).flatten().tolist()
    test_user_num = len(test_user)

    tmp_pre_R = np.sign(Estimated_R - gamma)
    tmp_test_R = np.sign(test_R - thre_rating)


    RL =np.sum(((tmp_test_R == 1) & (tmp_pre_R == 1)))
    RD =np.sum(((tmp_test_R == -1) & (tmp_pre_R == 1)))
    NL = np.sum(((tmp_test_R == 1) & (tmp_pre_R == -1)))
    ND= np.sum(((tmp_test_R == -1) & (tmp_pre_R == -1)))


    mis_cost = RD* cost_dict['RD']*b + NL* cost_dict['NL']*a+ RL* cost_dict['RL']*a + ND* cost_dict['ND']*b

    mis_cost = mis_cost / test_user_num

    return RL,RD,NL,ND,test_user_num,mis_cost


def three_way_decision_cost_rating(test_R, Estimated_R, cost_dict, thre_rating, a,b,alpha,beta):

    tmp_pre_R = np.zeros(Estimated_R.shape)
    tmp_pre_R[Estimated_R >= alpha] = 1
    tmp_pre_R[Estimated_R <= beta] = -1

    tmp_test_R = np.sign(test_R - thre_rating)

    is_test_user = test_R.sum(axis=1)
    is_test_user[is_test_user > 0] = 1
    is_test_user = is_test_user.reshape((len(is_test_user), 1)).flatten()

    test_user = np.argwhere(is_test_user != 0).flatten().tolist()
    test_user_num=len(test_user)

    RL = np.sum(((tmp_test_R == 1) & (tmp_pre_R == 1)))
    RD = np.sum(((tmp_test_R == -1) & (tmp_pre_R == 1)))
    BL = np.sum(((tmp_test_R == 1) & (tmp_pre_R == 0)))
    BD = np.sum(((tmp_test_R == -1) & (tmp_pre_R == 0)))
    NL = np.sum(((tmp_test_R == 1) & (tmp_pre_R == -1)))
    ND = np.sum(((tmp_test_R == -1) & (tmp_pre_R == -1)))


    mis_cost = RD* cost_dict['RD']*b + NL* cost_dict['NL']*a+ RL* cost_dict['RL']*a + ND* cost_dict['ND']*b+BL * cost_dict['BL']*a+BD * cost_dict['BD']*b

    total_cost=mis_cost/test_user_num

    return RL,RD, BL, BD, NL,ND,test_user_num,total_cost



def teaching_cost_CF(n_topic, num_user, num_item, theta):
    teaching_cost =theta * (pow(n_topic, 2)* num_user*num_item + pow(n_topic, 3)*num_user+pow(n_topic, 3)*num_item)
    return teaching_cost

def teaching_cost_CTR_LDA(n_topic, num_user, num_item, num_voca, theta):
    teaching_cost =theta *  (pow(n_topic, 2)* num_user*num_item + pow(n_topic, 3)*num_user+pow(n_topic, 3)*num_item+num_item*num_voca*n_topic)
    return teaching_cost

def integrate_precision_and_recall(data_name, data_path, method_data_path, integrate_result_path, min_hidden_neuron,
                                   max_hidden_neuron,step,test_folds, thre_rating):
    result_all_df = df()
    for n_topic in range(min_hidden_neuron, max_hidden_neuron + 1, step):
        info_dict = defaultdict()
        info_dict['n_topic'] = n_topic
        for test_fold in range(1, test_folds + 1):
            print("accuracy",n_topic, test_fold)
            if data_name == "citeulike-a":
                user_file = data_path + '/cf-a/' + 'cf-a-train-' + str(test_fold) + '-users.dat'
                item_file = data_path + '/cf-a/' + 'cf-a-train-' + str(test_fold) + '-items.dat'
                print (user_file)
                user_fp = open(user_file)
                print("OK")
                lines = user_fp.readlines()
                num_user = len(lines)
                user_fp.close()
                item_fp = open(item_file)
                lines = item_fp.readlines()
                num_item = len(lines)
                test_file = data_path + '/cf-a/' + 'cf-a-test-' + str(test_fold) + '-users.dat'
                results_path_data = method_data_path + '/cf-a-'

            elif data_name == "citeulike-t":
                user_file = data_path + '/cf-t/' + 'cf-t-train-' + str(test_fold) + '-users.dat'
                item_file = data_path + '/cf-t/' + 'cf-t-train-' + str(test_fold) + '-items.dat'
                print(user_file)
                user_fp = open(user_file)
                print("OK")
                lines = user_fp.readlines()
                num_user = len(lines)
                user_fp.close()
                item_fp = open(item_file)
                lines = item_fp.readlines()
                num_item = len(lines)
                test_file = data_path + '/cf-t/' + 'cf-t-test-' + str(test_fold) + '-users.dat'
                results_path_data = method_data_path + '/cf-t-'
            
            test_R, test_user_set, test_item_set, num_test_ratings, test_C = data_process.read_user(test_file, num_user,num_item)
            test_mask_R = test_R
            results_path_sub = results_path_data + str(n_topic) + '-' + str(test_fold) + '/'
            u_file = results_path_sub + 'final-U.dat'
            v_file = results_path_sub + 'final-V.dat'
            print (u_file,v_file)
            U = np.mat(np.loadtxt(u_file, dtype=np.float))
            V = np.mat(np.loadtxt(v_file, dtype=np.float))
            estimated_R = U * V.T
            print ("read_finish")
            for n_top in range(50, 301, 50):
                Recall = recall_n_top(test_R, test_mask_R, estimated_R, num_test_ratings, thre_rating, n_top)
                Precision = precision_n_top(test_R, test_mask_R, estimated_R, num_test_ratings, thre_rating, n_top)
                if test_fold == 1:
                    info_dict['n_top_' + str(n_top) + '_Recall'] = Recall / test_folds
                    info_dict['n_top_' + str(n_top) + '_Precision'] = Precision / test_folds
                else:
                    info_dict['n_top_' + str(n_top) + '_Recall'] += Recall / test_folds
                    info_dict['n_top_' + str(n_top) + '_Precision'] += Precision / test_folds
                print(n_top, Recall, Precision)

        if len(result_all_df) == 0:
            result_all_df = df(info_dict, index=[0])
        else:
            result_one_df = df(info_dict, index=[0])
            result_all_df = pd.concat([result_all_df, result_one_df], ignore_index=True)
    result_all_df.to_csv(integrate_result_path +'/'+data_name + "_results.csv")
    time.sleep(2)


def integrate_decision_and_teaching_cost(data_name, data_path, method_name,method_data_path, integrate_result_path,
                                         min_hidden_neuron,max_hidden_neuron, step, test_folds, cost_dict_list,theta,thre_rating, a, b):
    two_way_info_df = df(
        columns=['n_topic', 'test_fold', 'cost_num', 'RL', 'RD', 'NL', 'ND', 'd_cost', 'test_user_num', 't_cost'])
    three_way_info_df = df(
        columns=['n_topic', 'test_fold', 'cost_num', 'RL', 'RD', 'BL', 'BD', 'NL', 'ND', 'd_cost', 'test_user_num',
                 't_cost'])
    for n_topic in range(min_hidden_neuron, max_hidden_neuron + 1, step):
        for test_fold in range(1, test_folds + 1):
            print("cost",n_topic, test_fold)

            info_dict_two = defaultdict()
            info_dict_two['n_topic'] = n_topic

            info_dict_three = defaultdict()
            info_dict_three['n_topic'] = n_topic

            info_dict_two['test_fold'] = test_fold
            info_dict_three['test_fold'] = test_fold

            if data_name == "citeulike-a":

                test_file = data_path + '/cf-a/' + 'cf-a-test-' + str(test_fold) + '-users.dat'
                user_file = data_path + '/cf-a/' + 'cf-a-train-' + str(test_fold) + '-users.dat'
                item_file = data_path + '/cf-a/' + 'cf-a-train-' + str(test_fold) + '-items.dat'
                user_fp = open(user_file)
                lines = user_fp.readlines()
                num_user = len(lines)
                user_fp.close()
                item_fp = open(item_file)
                lines = item_fp.readlines()
                num_item = len(lines)
                num_voca = 8000
                results_path_data = method_data_path + '/cf-a-'

            elif data_name == "citeulike-t":

                test_file = data_path + '/cf-t/' + 'cf-t-test-' + str(test_fold) + '-users.dat'
                user_file = data_path + '/cf-t/' + 'cf-t-train-' + str(test_fold) + '-users.dat'
                item_file = data_path + '/cf-t/' + 'cf-t-train-' + str(test_fold) + '-items.dat'
                user_fp = open(user_file)
                lines = user_fp.readlines()
                num_user = len(lines)
                user_fp.close()
                item_fp = open(item_file)
                lines = item_fp.readlines()
                num_item = len(lines)
                num_voca = 20000
                results_path_data = method_data_path + '/cf-t-'

            test_R, test_user_set, test_item_set, num_test_ratings, test_C = data_process.read_user(test_file, num_user,num_item)
            results_path_sub = results_path_data + str(n_topic) + '-' + str(test_fold) + '/'

            u_file = results_path_sub + 'final-U.dat'
            v_file = results_path_sub + 'final-V.dat'
            U = np.mat(np.loadtxt(u_file, dtype=np.float))
            V = np.mat(np.loadtxt(v_file, dtype=np.float))
            estimated_R = U * V.T
            cost_num = 1
            for cost_dict in cost_dict_list:
                info_dict_two['cost_num'] = cost_num
                info_dict_three['cost_num'] = cost_num
                cost_num = cost_num + 1

                alpha = (cost_dict['RD'] - cost_dict['BD']) / (
                            (cost_dict['RD'] - cost_dict['BD']) + (cost_dict['BL'] - cost_dict['RL']))
                beta = (cost_dict['BD'] - cost_dict['ND']) / (
                            (cost_dict['BD'] - cost_dict['ND']) + (cost_dict['NL'] - cost_dict['BL']))
                gamma = (cost_dict['RD'] - cost_dict['ND']) / (
                            (cost_dict['RD'] - cost_dict['ND']) + (cost_dict['NL'] - cost_dict['RL']))

                # d_cost represents decision cost
                info_dict_two['RL'], info_dict_two['RD'], info_dict_two['NL'], info_dict_two['ND'], info_dict_two[
                    'test_user_num'], info_dict_two['d_cost'] = \
                    two_way_decision_cost_rating(test_R, estimated_R, cost_dict, thre_rating, a, b, gamma)

                info_dict_three['RL'], info_dict_three['RD'], info_dict_three['BL'], info_dict_three['BD'], \
                info_dict_three['NL'], info_dict_three['ND'], info_dict_three['test_user_num'], info_dict_three[
                    'd_cost'] = \
                    three_way_decision_cost_rating(test_R, estimated_R, cost_dict, thre_rating, a, b, alpha, beta)
                print('calculate time cost')
                if method_name == 'CF':
                    print ('t_cost',method_name)
                    t_cost = teaching_cost_CF(n_topic, num_user, num_item, theta)
                else:
                    print ('t_cost',method_name)
                    t_cost=teaching_cost_CTR_LDA(n_topic, num_user, num_item, num_voca, theta)
                info_dict_two['t_cost'] = t_cost
                info_dict_three['t_cost'] = t_cost

                two_way_info_df = pd.concat([two_way_info_df, df(info_dict_two, index=[0])], ignore_index=True)
                three_way_info_df = pd.concat([three_way_info_df, df(info_dict_three, index=[0])], ignore_index=True)

    two_way_result_file = integrate_result_path + '/'+data_name + "_two_way_info.csv"
    three_way_result_file = integrate_result_path +'/'+ data_name + "_three_way_info.csv"
    two_way_info_df.to_csv(two_way_result_file, index=False,
                           columns=['n_topic', 'test_fold', 'cost_num', 'RL', 'RD', 'NL', 'ND', 'd_cost',
                                    'test_user_num', 't_cost'])
    three_way_info_df.to_csv(three_way_result_file, index=False,
                             columns=['n_topic', 'test_fold', 'cost_num', 'RL', 'RD', 'BL', 'BD', 'NL', 'ND', 'd_cost',
                                      'test_user_num', 't_cost'])
    time.sleep(2)


if __name__=="__main__":

    recall_avg = 0
    precision_avg = 0
    accuracy_avg = 0
    n = 0
    test_folds = 20
    data_list = ["citeulike-a", "citeulike-t"]
    method_name_list=['CF','LDA','CTR']
    num_processes=6
    theta=1e-9

    # the dimensionality of the output in the first layer
    first_hidden_nueron=200
    thre_rating = 0.5
    a=1
    b=0.01
    cost_dict= {'RL': 0, 'RD': 10, 'BL': 2, 'BD': 2, 'NL': 5, 'ND': 0}
    cost_dict_list=[cost_dict]
    data_path='../data'
    integrate_path='../results'
    if not os.path.exists(integrate_path):
        os.mkdir(integrate_path)
    pool = Pool(processes=num_processes)
    
    for data_name in data_list:
        if data_name=="citeulike-a":
            max_hidden_neuron = 100
            min_hidden_neuron=5
            step=5
        else:
            max_hidden_neuron = 49
            min_hidden_neuron=3
            step=2
        for method_name in method_name_list:
            method_data_path =os.path.join(data_path,method_name+"_results")
            integrate_result_path =os.path.join(integrate_path, method_name)
            if not os.path.exists(integrate_result_path):
                os.mkdir(integrate_result_path)
            pool.apply_async(integrate_precision_and_recall, (
            data_name, data_path, method_data_path, integrate_result_path, min_hidden_neuron,max_hidden_neuron,step, test_folds, thre_rating,))
            pool.apply_async(integrate_decision_and_teaching_cost, (
            data_name, data_path, method_name,method_data_path, integrate_result_path, min_hidden_neuron,max_hidden_neuron,step, test_folds, cost_dict_list, theta, thre_rating, a, b,))
    pool.close()
    pool.join()










