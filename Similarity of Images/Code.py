#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:56:22 2018

@author: varshath
"""

import numpy as np
import pandas as pd

import csv
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold


def splitData(data,offset,percent):
    dataSize=int(len(data)*percent/100)
    dataSplit=data[offset:offset+dataSize+1]
    offset+=dataSize
    return dataSplit,offset





def read_file(fileName):
    df=pd.read_csv(fileName,nrows=1500)
    return df  
    
def extract_features_concat(feature_file,pairs):
    join=pd.merge(pairs, feature_file, left_on='img_id_A', right_on='img_id')
    res=(pd.merge(join, feature_file, left_on='img_id_B', right_on='img_id'))
    res=res.drop(['img_id_x','img_id_B', 'img_id_A', 'img_id_y'], axis=1)
    return res
    


def extract_features_sub(feature_file,pairs,cols):
    join1=pd.merge(pairs, feature_file, left_on='img_id_A', right_on='img_id')
    join2=pd.merge(pairs, feature_file, left_on='img_id_B', right_on='img_id')
    join1=join1.drop(join1.columns[cols], axis=1)
    join2=join2.drop(join2.columns[cols], axis=1)
    join1=join1.set_index('target')
    join2=join2.set_index('target')
    features=abs(join1.sub(join2))
    features=features.reset_index()
    
    return features
 

def preprocess_gsc():
    feature_file=read_file("GSC-Features-Data/GSC-Features.csv")
    same_pairs=read_file("GSC-Features-Data/same_pairs.csv")
    diff_pairs=read_file("GSC-Features-Data/diffn_pairs.csv")
    same_pairs_features_concat=extract_features_concat(feature_file,same_pairs)
    diff_pairs_features_concat=extract_features_concat(feature_file,diff_pairs)
    features_concat=same_pairs_features_concat.append(diff_pairs_features_concat)
    cols = [0,1,3]
    same_pairs_features_sub=extract_features_sub(feature_file,same_pairs,cols)
    diff_pairs_features_sub=extract_features_sub(feature_file,diff_pairs,cols)
    features_sub=same_pairs_features_sub.append(diff_pairs_features_sub)
    features_concat=features_concat.sample(frac=1).reset_index(drop=True)
    features_sub=features_sub.sample(frac=1).reset_index(drop=True)
    return features_concat,features_sub
    
def preprocess_human_observed():
    feature_file=read_file("HumanObserved-Features-Data/HumanObserved-Features-Data.csv")
    feature_file=feature_file.drop(feature_file.columns[0],axis=1)
    same_pairs=read_file("HumanObserved-Features-Data/same_pairs.csv")
    diff_pairs=read_file("HumanObserved-Features-Data/diffn_pairs.csv")
    same_pairs_features_concat=extract_features_concat(feature_file,same_pairs)
    diff_pairs_features_concat=extract_features_concat(feature_file,diff_pairs)
    features_concat=same_pairs_features_concat.append(diff_pairs_features_concat)
    cols = [0,1,3]
    same_pairs_features_sub=extract_features_sub(feature_file,same_pairs,cols)
    diff_pairs_features_sub=extract_features_sub(feature_file,diff_pairs,cols)
    features_sub=same_pairs_features_sub.append(diff_pairs_features_sub)
    features_concat=features_concat.sample(frac=1).reset_index(drop=True)
    features_sub=features_sub.sample(frac=1).reset_index(drop=True)
    return features_concat,features_sub
    
    

gsc_features_concat,gsc_features_sub=preprocess_gsc()
human_features_concat,human_features_sub=preprocess_human_observed()

