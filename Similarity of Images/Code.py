#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:50:01 2018

@author: varshath
"""

import numpy as np
import pandas as pd


    
    
def extract_features_gsc_concat(same_pairs,targetValue):
    feature_writer_1=['f'+str(i) for i in range(1, 513)]
    feature_writer_2=['f'+str(i) for i in range(513, 1025)]
    features=[]
    for index,row in same_pairs.iterrows():
        if index>100:
            break
        writer_a=row['img_id_A']
        writer_b=row['img_id_B']
        features_a=feature_file.loc[feature_file['img_id'] == writer_a].drop(['img_id'], axis=1)
        features_b=feature_file.loc[feature_file['img_id'] == writer_b].drop(['img_id'], axis=1)
        features_a.columns=feature_writer_1
        features_a.reset_index(drop=True, inplace=True)
        features_b.columns=feature_writer_2
        features_b.reset_index(drop=True, inplace=True)
        concatenated_features=pd.concat([features_a, features_b],axis=1)
        if index<1:
            features=concatenated_features
        else:
            features=features.append(concatenated_features)
    features['target']=targetValue
    return features


def extract_features_gsc_sub(same_pairs,targetValue):
    features=[]
    for index,row in same_pairs.iterrows():
        if index>1:
            break
        writer_a=row['img_id_A']
        writer_b=row['img_id_B']
        features_a=feature_file.loc[feature_file['img_id'] == writer_a].drop(['img_id'], axis=1)
        features_b=feature_file.loc[feature_file['img_id'] == writer_b].drop(['img_id'], axis=1)
        return abs(features_a.values - features_b.values)
        '''
        concatenated_features=pd.concat([features_a, features_b],axis=1)
        if index<1:
            features=concatenated_features
        else:
            features=features.append(concatenated_features)
    features['target']=targetValue
    return features
        '''


def read_file(fileName):
    df=pd.read_csv(fileName)
    return df 

feature_file=read_file("GSC-Features-Data/GSC-Features.csv")
same_pairs=read_file("GSC-Features-Data/same_pairs.csv")
diff_pairs=read_file("GSC-Features-Data/diffn_pairs.csv")
same_pairs_features=extract_features_gsc_concat(same_pairs,1)
diff_pairs_features=extract_features_gsc_concat(diff_pairs,0)
x=extract_features_gsc_sub(diff_pairs,0)
 
'''    
writers_a = features['img_id_A']
writers_b=features['img_id_B']
targets=features['target']
features.drop(labels=['img_id_A','img_id_B','target'], axis=1,inplace = True)
features.insert(0, 'img_id_B', writers_b)
features.insert(0, 'img_id_A', writers_a)
features.insert(2, 'target', targets)
'''




