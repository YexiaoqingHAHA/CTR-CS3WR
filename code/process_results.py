# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:30:59 2019

@author: Txin
"""

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import os 


if __name__=='__main__':
    method_list=['CF','LDA','CTR']
    dataset_list=['citeulike-a','citeulike-t']
    for method in method_list:
        data_path=os.path.join('../results',method)
        for  dataset in dataset_list:
            if dataset=='citeulike-a':
                theta=0.01
            elif dataset=='citeulike-t':
                theta=0.3
            ##Three_way_recommendation#####
            three_way_file_name=dataset+'_three_way_info.csv'
            three_way_file=os.path.join(data_path,three_way_file_name)
            three_result_info_df = pd.read_csv(three_way_file)
            three_group=three_result_info_df.groupby(['n_topic','cost_num']).mean()
            three_group.reset_index(inplace=True)
            three_group[['RL', 'RD', 'BL', 'BD', 'NL', 'ND','test_user_num']]=three_group[['RL', 'RD', 'BL', 'BD', 'NL', 'ND','test_user_num']].astype(int)
            
            three_bou_num=df()
            three_cost_info=df()
            
            three_bou_num['n_topic']=three_group['n_topic']
            num_name='num_bou'
            three_bou_num[num_name] = three_group['BL'] + three_group['BD']
            cost_info_df=three_group[['n_topic','d_cost','t_cost']]
            cost_info_df.reset_index(inplace=True,drop=True)
            d_name='cost_d'
            t_name='cost_t'
            three_cost_info['n_topic']=cost_info_df['n_topic']
            three_cost_info[d_name]=cost_info_df['d_cost']
            three_cost_info[t_name]=cost_info_df['t_cost']*theta
            
            three_bou_file_name=dataset+'_three_way_bou_num.csv'
            three_bou_file=os.path.join(data_path,three_bou_file_name)
            three_bou_num.to_csv(three_bou_file,index=None)
            three_cost_file_name=dataset+'_three_way_cost.csv'
            three_cost_file=os.path.join(data_path,three_cost_file_name)
            three_cost_info.to_csv(three_cost_file,index=None)
            
            ###Two_way_recommendation
            two_way_file_name=dataset+'_two_way_info.csv'
            two_way_file=os.path.join(data_path,two_way_file_name)
            two_result_info_df = pd.read_csv(two_way_file)
            two_groups=two_result_info_df.groupby(['n_topic','cost_num']).mean()
            two_groups.reset_index(inplace=True)
            two_groups[['RL', 'RD', 'NL', 'ND','test_user_num']]=two_groups[['RL', 'RD', 'NL', 'ND','test_user_num']].astype(int)

            two_bou_num=df()
            two_cost_info=df()
           
            two_cost_info['n_topic']=cost_info_df['n_topic']
            two_cost_info[d_name]=cost_info_df['d_cost']
            two_cost_info[t_name]=cost_info_df['t_cost']*theta
          
            two_cost_file_name=dataset+'_two_way_cost.csv'
            two_cost_file=os.path.join(data_path,two_cost_file_name)
            two_cost_info.to_csv(two_cost_file,index=None)
