# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 21:24:29 2018

@author: Xiaoqing Ye
"""

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib

from matplotlib import pyplot as plt
import os

max_hidden_neuron = 50
picture_path='picture/1/'
two_result_file="results/citeulike-a_two_way_info.csv"
three_result_file="results/citeulike-a_three_way_info.csv"
max_n=int(max_hidden_neuron/5)


###error rate of two_way and three_way
if os.path.exists(picture_path) == False:
    os.mkdir(picture_path)

two_result_info_df = pd.read_csv(two_result_file)
two_result_info_df = two_result_info_df[(two_result_info_df['test_fold'] ==1) & (two_result_info_df['cost_num']==1)]
   
three_result_info_df = pd.read_csv(three_result_file)

three_result_info_df = three_result_info_df[(three_result_info_df['test_fold'] ==1) & (three_result_info_df['cost_num']==1)]
x = three_result_info_df['n_topic']

###error rate of two_way and three_way
two_error_ratio=two_result_info_df['RL']/(two_result_info_df['RL']+two_result_info_df['RD'])
three_error_ratio=three_result_info_df['RL']/(three_result_info_df['RL']+three_result_info_df['RD'])
fig=plt.figure(figsize=(6,6))
plt.plot(x,two_error_ratio)
plt.plot(x,three_error_ratio)
plt.show()
plt.savefig(picture_path + 'precision_rate.png')
                                

###the number variation of three region
if os.path.exists(picture_path) == False:
    os.mkdir(picture_path)

two_result_info_df = pd.read_csv(two_result_file)
two_result_info_df = two_result_info_df[(two_result_info_df['test_fold'] ==1) & (two_result_info_df['cost_num']==1)]
   
three_result_info_df = pd.read_csv(three_result_file)
three_result_info_df = three_result_info_df[(three_result_info_df['test_fold'] ==1) & (three_result_info_df['cost_num']==1)]
x = three_result_info_df['n_topic']

pos = three_result_info_df['RL'] + three_result_info_df['RD']
bon = three_result_info_df['BL'] + three_result_info_df['BD']
neg = three_result_info_df['NL'] + three_result_info_df['ND']
total=pos+bon+neg

fig = plt.figure(figsize=(6, 6))
plt.bar(x, pos)
plt.show()
plt.savefig(picture_path + 'pos_num.png')

fig = plt.figure(figsize=(6, 6))
plt.bar(x, bon)
plt.show()
plt.savefig(picture_path + 'bon_num.png')

fig = plt.figure(figsize=(6, 6))
plt.bar(x, neg)
plt.show()
plt.savefig(picture_path + 'neg_num.png')

fig = plt.figure(figsize=(6, 6))
plt.bar(x, pos)
plt.bar(x, bon, bottom=pos)
plt.bar(x, neg, bottom=pos+bon)
plt.show()
plt.savefig(picture_path + 'all_num.png')



###teching_cost, decision_cost, total_cost

if os.path.exists(picture_path) == False:
    os.mkdir(picture_path)
    
two_result_info = pd.read_csv(two_result_file)
two_result_info = two_result_info[two_result_info['cost_num']==2]
two_result_info_df = two_result_info.groupby('n_topic').mean()
two_result_info_df.drop('test_fold', axis=1, inplace=True)
two_result_info_df[['RL', 'RD', 'NL', 'ND']] = two_result_info_df[['RL', 'RD', 'NL', 'ND']].astype(int)
    
    

three_result_info = pd.read_csv(three_result_file)
three_result_info = three_result_info[three_result_info['cost_num']==2]
three_result_info_df = three_result_info.groupby('n_topic').mean()
three_result_info_df.drop('test_fold', axis=1, inplace=True)
three_result_info_df[['RL', 'RD', 'BL', 'BD', 'NL', 'ND']] = three_result_info_df[['RL', 'RD', 'BL', 'BD', 'NL', 'ND']].astype(int)


max_n=20
fig = plt.figure(figsize=(30, 6))
x = three_result_info_df.index.values[:max_n]
two_d_cost_array=two_result_info_df['d_cost'].values[:max_n]
three_d_cost_array=three_result_info_df['d_cost'].values[:max_n]
t_cost_array=three_result_info_df['t_cost'].values[:max_n]*0.01
for epsilon in range(0, 11, 1):
    
    two_d_cost = 0.1 * epsilon *two_d_cost_array
    three_d_cost = 0.1 * epsilon *three_d_cost_array
    t_cost = (1 - 0.1 * epsilon) * t_cost_array

    two_total_cost_array = two_d_cost + t_cost
    three_total_cost_array = three_d_cost + t_cost
    two_d_cost = two_d_cost.tolist()
    three_d_cost = three_d_cost.tolist()
    t_cost = t_cost.tolist()
    two_total_cost= two_total_cost_array.tolist()
    three_total_cost= three_total_cost_array.tolist()


    plt.subplot(2, 11, epsilon + 1)
    plt.plot(x, two_d_cost,label='two_d')
    plt.plot(x, three_d_cost,label='three_d')
    plt.plot(x, t_cost,label='teach')
    plt.legend()

    plt.subplot(2, 11, epsilon + 1 + 11)
    plt.plot(x, two_total_cost,label='two_total')
    plt.plot(x, three_total_cost,label='three_total')
    plt.legend()

plt.show()
plt.savefig(picture_path + 'total_cost.png')



