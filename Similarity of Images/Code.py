#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:50:01 2018

@author: varshath
"""

import numpy as np
import pandas as pd


    
    
def extract_features_concat(feature_file,same_pairs,targetValue,feature_length):
    feature_writer_1=['f'+str(i) for i in range(1, feature_length)]
    x=feature_length
    feature_writer_2=['f'+str(i) for i in range(feature_length+1, x*2)]
    features=[]
    for index,row in same_pairs.iterrows():
        if index>10:
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
    return features.values


def extract_features_gsc_sub(feature_file,same_pairs,targetValue):
    features=[]
    for index,row in same_pairs.iterrows():
        if index>10:
            break
        writer_a=row['img_id_A']
        writer_b=row['img_id_B']
        features_a=feature_file.loc[feature_file['img_id'] == writer_a].drop(['img_id'], axis=1)
        features_b=feature_file.loc[feature_file['img_id'] == writer_b].drop(['img_id'], axis=1)
        sub_features=abs(features_a.values - features_b.values)
        if index<1:
            features=sub_features
        else:
            features=np.concatenate((features,sub_features),axis=0)
    target=[[targetValue for i in range(1)] for j in range(len(features))]
    features = np.concatenate((features, target), 1)
    return features


def read_file(fileName):
    df=pd.read_csv(fileName)
    return df 

def preprocess_gsc():
    feature_file=read_file("GSC-Features-Data/GSC-Features.csv")
    same_pairs=read_file("GSC-Features-Data/same_pairs.csv")
    diff_pairs=read_file("GSC-Features-Data/diffn_pairs.csv")
    same_pairs_features_concat=extract_features_concat(feature_file,same_pairs,1,513)
    diff_pairs_features_concat=extract_features_concat(feature_file,diff_pairs,0,513)
    same_pairs_features_sub=np.array(extract_features_gsc_sub(feature_file,same_pairs,1))
    diff_pairs_features_sub=np.array(extract_features_gsc_sub(feature_file,diff_pairs,0))
    return same_pairs_features_concat,diff_pairs_features_concat,same_pairs_features_sub,diff_pairs_features_sub

def preprocess_human_observed():
    feature_file=read_file("HumanObserved-Features-Data/HumanObserved-Features-Data.csv")
    feature_file=feature_file.drop(feature_file.columns[0],axis=1)
    same_pairs=read_file("HumanObserved-Features-Data/same_pairs.csv")
    diff_pairs=read_file("HumanObserved-Features-Data/diffn_pairs.csv")
    same_pairs_features_concat=extract_features_concat(feature_file,same_pairs,1,10)
    diff_pairs_features_concat=extract_features_concat(feature_file,diff_pairs,0,10)
    same_pairs_features_sub=np.array(extract_features_gsc_sub(feature_file,same_pairs,1))
    diff_pairs_features_sub=np.array(extract_features_gsc_sub(feature_file,diff_pairs,0))
    return same_pairs_features_concat,diff_pairs_features_concat,same_pairs_features_sub,diff_pairs_features_sub

gsc_same_pairs_features_concat,gsc_diff_pairs_features_concat,gsc_same_pairs_features_sub,gsc_diff_pairs_features_sub=preprocess_gsc()
human_same_pairs_features_concat,human_diff_pairs_features_concat,human_same_pairs_features_sub,human_diff_pairs_features_sub=preprocess_human_observed()
