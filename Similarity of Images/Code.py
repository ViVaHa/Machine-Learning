#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 21:10:09 2018

@author: varshath
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:56:22 2018

@author: varshath
"""

import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop,Adam
import tensorflow as tf 


#for reading a file
def read_file(fileName,size):
    df=pd.read_csv(fileName,nrows=size)
    return df  
    

#this method is for concatenating the features
def extract_features_concat(feature_file,pairs):
    join=pd.merge(pairs, feature_file, left_on='img_id_A', right_on='img_id')
    res=(pd.merge(join, feature_file, left_on='img_id_B', right_on='img_id'))
    res=res.drop(['img_id_x','img_id_B', 'img_id_A', 'img_id_y'], axis=1)
    return res

#this method is for subtraction of features    
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
 
    

#preprocessing GSC data set
def preprocess_gsc(size):
    feature_file=pd.read_csv("GSC-Features-Data/GSC-Features.csv")
    same_pairs=read_file("GSC-Features-Data/same_pairs.csv",size)
    diff_pairs=read_file("GSC-Features-Data/diffn_pairs.csv",size)
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
 
    
#preprocessing human observed data set
def preprocess_human_observed(size):
    feature_file=pd.read_csv("HumanObserved-Features-Data/HumanObserved-Features-Data.csv")
    feature_file=feature_file.drop(feature_file.columns[0],axis=1)
    same_pairs=read_file("HumanObserved-Features-Data/same_pairs.csv",size)
    diff_pairs=read_file("HumanObserved-Features-Data/diffn_pairs.csv",size)
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
 
#method to split data for training,validation,testing  
def splitData(data,offset,percent):
    dataSize=int(len(data)*percent/100)
    dataSplit=data[offset:offset+dataSize]
    offset+=dataSize
    return dataSplit,offset


#bigSigma for linear regression
def getBigSigma(data):
    selector = VarianceThreshold()
    selector.fit_transform(data)
    varianceVector=selector.variances_
    bigSigma=np.zeros((len(data[0]),len(data[0])))
    for i in range(len(data[0])):
        bigSigma[i][i]=varianceVector[i]+0.2
    bigSigma=np.dot(bigSigma,100)
    return bigSigma


def getMean(data,numOfClusters):
    kmeans = KMeans(n_clusters=numOfClusters, random_state=5).fit(data)
    mean=kmeans.cluster_centers_
    return mean

def findScalarVal(data,mean,bigSigmaInverse):
    X=np.subtract(data,mean)
    Y=np.dot(X,bigSigmaInverse)
    Z=np.dot(Y,np.transpose(X))
    return math.exp(-0.5*Z)


#Method to get PHI matrix
def getDesignMatrix(data,mean,bigSigma):
    bigSigmaInverse=np.linalg.inv(bigSigma)
    designMatrix=np.zeros((len(data),len(mean)))
    for i in range(len(data)):
        for j in range(len(mean)):
            dataRow=data[i]
            m=mean[j]
            designMatrix[i][j]=findScalarVal(dataRow,m,bigSigmaInverse)
    return designMatrix


#Method for erms calculation of linear regression
def erms(predictedValues,actualValues):
    totalSum=0
    correctPrediction=0
    for i in range(len(predictedValues)):
        diff=(actualValues[i]-predictedValues[i])**2
        totalSum+=diff
        if (int(predictedValues[i])==actualValues[i]):
            correctPrediction+=1
    percent=correctPrediction*100/len(actualValues)
    rms=math.sqrt(totalSum/len(actualValues))
    return percent,rms
 
    
#seperating target Vector from features
def get_target_and_features(data):
    target=data['target']
    data=data.drop(data.columns[0],axis=1)
    return data,np.array(target)


#Sigmoid fn for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))



#Method to predict accuracy for logistic regression
def logistic_accuracy(h,y):
    c=0
    for i in range(len(h)):
        if int(np.around(h[i],0))==y[i]:
            c+=1
    return c*100/len(y)
            


#Linear regression for each dataset
def linear_regression(data):
    inputFeatures,targetVector=get_target_and_features(data)
    
    #inputFeatures=inputFeatures.loc[:, (inputFeatures != inputFeatures.iloc[0]).any()] 
    inputFeatures=inputFeatures.sample(frac=1).reset_index(drop=True)
    inputFeatures=inputFeatures.values
    offset=0
    trainData,offset=splitData(inputFeatures,offset,70)
    validationData,offset=splitData(inputFeatures,offset,15)
    testData=inputFeatures[offset:]
    offset=0
    trainTargetVector,offset=splitData(targetVector,offset,70)
    validationTargetVector,offset=splitData(targetVector,offset,15)
    testTargetVector=targetVector[offset:]
    num_of_clusters=10
    mean=getMean(trainData,num_of_clusters)
    bigSigma=getBigSigma(trainData)
    
    
    
    
    
    trainDesignMatrix=getDesignMatrix(trainData,mean,bigSigma)
    validationDesignMatrix=getDesignMatrix(validationData,mean,bigSigma)
    testDesignMatrix=getDesignMatrix(testData,mean,bigSigma)
    
    
    
    
    
    
    
    
    
    
    
    num_of_features=num_of_clusters
    currentWeights=np.ones((num_of_features, 1))
    learningRate=0.001
    w=[]
    regularizationTerm=0.4
    trainingAccuracies=[]
    training_erms=[]
    validationAccuracies=[]
    validation_erms=[]
    testing_erms=[]
    testingAccuracies=[]
    epochs=1000
    for i in range(epochs): 
        deltaED= -np.dot((trainTargetVector[i] - np.dot(np.transpose(currentWeights),trainDesignMatrix[i])),np.reshape(trainDesignMatrix[i],(1,num_of_features)))
        deltaEW=np.dot(currentWeights,regularizationTerm)
        deltaE=np.add(np.reshape(deltaED,(num_of_features,1)),deltaEW)
        deltaW=-np.dot(deltaE,(learningRate))
        newWeights=currentWeights+deltaW
        w.append(newWeights)
        currentWeights=newWeights
        
        
        predictedTrainingVals=np.dot(np.transpose(newWeights),np.transpose(trainDesignMatrix))
        trainingAccuracy,train_erms=erms(np.transpose(predictedTrainingVals),trainTargetVector)
        
        trainingAccuracies.append(trainingAccuracy) 
        training_erms.append(train_erms)
        
        
        validationPredictedVals=np.dot(np.transpose(newWeights),np.transpose(validationDesignMatrix))
        validationAccuracy,valid_erms=erms(np.transpose(validationPredictedVals),validationTargetVector)
        validationAccuracies.append(validationAccuracy)
        validation_erms.append(valid_erms)
        
        
        testPredictedVals=np.dot(np.transpose(newWeights),np.transpose(testDesignMatrix))
        testAccuracy,test_erms=erms(np.transpose(testPredictedVals),testTargetVector)
        testingAccuracies.append(testAccuracy)    
        testing_erms.append(test_erms)
        
    
    plt.plot(range(epochs), training_erms, label = 'Training')
    plt.plot(range(epochs), validation_erms, label = 'Validation')
    plt.plot(range(epochs), testing_erms, label = 'Testing')
    
    
    plt.legend()
    plt.ylabel('Erms')
    plt.xlabel('Iterations')
    #plt.savefig("linear_reg_"+str(li_count)+".jpg")
    plt.show()
    
    print ('----------Gradient Descent Solution(Linear Regression)--------------------')
    print ("E_rms Training   = " + str(np.around(min(training_erms),5)))
    print ("E_rms Valdiation   = " + str(np.around(min(validation_erms),5)))
    print ("E_rms Testing   = " + str(np.around(min(testing_erms),5)))
    


#logistic regression for each data set
def logistic_regression(data,learning_rate=0.001):

    inputFeatures,targetVector=get_target_and_features(data)
    inputFeatures=inputFeatures.loc[:, (inputFeatures != inputFeatures.iloc[0]).any()] 
    #inputFeatures=inputFeatures.sample(frac=1).reset_index(drop=True)
    inputFeatures=inputFeatures.values
    offset=0
    trainData,offset=splitData(inputFeatures,offset,80)
    validationData,offset=splitData(inputFeatures,offset,10)
    testData=inputFeatures[offset:]
    offset=0
    trainTargetVector,offset=splitData(targetVector,offset,80)
    validationTargetVector,offset=splitData(targetVector,offset,10)
    testTargetVector=targetVector[offset:]    
    
    
    
    
    
    
    current_weights = np.random.rand(trainData.shape[1],1)
    
    trainingAccuracies=[]
    validationAccuracies=[]
    testingAccuracies=[]
    epochs=800
    #print(currentWeights.shape)
    #print(trainData.shape)
    gradients=[]
    
    trainData=trainData.T
    
    print(trainData.shape)
    print(current_weights.shape)
    y=np.mat(trainTargetVector).T
    v_y=np.mat(validationTargetVector).T
    t_y=np.mat(testTargetVector).T
    
    
    for i in range(epochs): 
        z = np.dot(np.transpose(current_weights),trainData)
        h = sigmoid(z).transpose()
        gradient = np.dot(trainData, (h - y)) / y.size
        gradients.append(gradient)
        current_weights -= gradient * learning_rate
        trainingAccuracy=logistic_accuracy(h,y)
        
        
        
        
        
        trainingAccuracies.append(trainingAccuracy) 
        #training_erms.append(train_erms)
        
        v_z=np.dot(np.transpose(current_weights),validationData.T)
        v_h=sigmoid(v_z).T
        validationAccuracy=logistic_accuracy(v_h,v_y)
        t_z=np.dot(np.transpose(current_weights),testData.T)
        t_h=sigmoid(t_z).T
        testAccuracy=logistic_accuracy(t_h,t_y)
        validationAccuracies.append(validationAccuracy)
        testingAccuracies.append(testAccuracy)
        
        
        '''
        validationPredictedVals=sigmoid(np.dot(validationData, np.transpose(current_weights)))
        validationAccuracy,valid_erms=cost_fn(np.transpose(validationPredictedVals),validationTargetVector)
        validationAccuracies.append(validationAccuracy)
        validation_erms.append(valid_erms)
        
        
        testPredictedVals=sigmoid(np.dot(testData, np.transpose(current_weights)))
        testAccuracy,test_erms=cost_fn(np.transpose(testPredictedVals),testTargetVector)
        testingAccuracies.append(testAccuracy)    
        testing_erms.append(test_erms)
        
        '''
    plt.plot(range(epochs), trainingAccuracies, label = 'Training')
    plt.plot(range(epochs), validationAccuracies, label = 'Validation')
    plt.plot(range(epochs), testingAccuracies, label = 'Testing')
    
    
    plt.legend()
    plt.ylabel('Accuracies')
    plt.xlabel('Iterations')
    #plt.savefig("logisitc_reg_"+str(lo_count)+".jpg")
    plt.show()
    
    print ('----------Logistic Regression Solution--------------------')
    print ("Accuracy Training   = " + str(np.around(max(trainingAccuracies),5)))
    print ("Accuracy Valdiation   = " + str(np.around(max(validationAccuracies),5)))
    print ("Accuracy Testing   = " + str(np.around(max(testingAccuracies),5)))
    
    #print(max(testPredictedVals))




#neural network using keras for each data set
def neural_network(data):

    inputFeatures,targetVector=get_target_and_features(data)
    inputFeatures=inputFeatures.loc[:, (inputFeatures != inputFeatures.iloc[0]).any()] 
    #inputFeatures=inputFeatures.sample(frac=1).reset_index(drop=True)
    inputFeatures=inputFeatures.values
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(inputFeatures, targetVector, test_size=0.25, random_state=42)
    
    
    input_size = len(inputFeatures[0])
    drop_out = 0.5
    first_dense_layer_nodes  = 100
    second_dense_layer_nodes = 2
    
    validation_data_split = 0.20
    num_epochs = 1000
    model_batch_size = 1024
    early_patience = 100
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
    
    
    
    
    
    
    
    
    
    model = Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('sigmoid'))
    opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    
    
    
    
    history = model.fit(X_train
                        , np_utils.to_categorical(np.array(y_train), 2)
                        , validation_split=validation_data_split
                        , epochs=num_epochs
                        , batch_size=model_batch_size
                        , callbacks = [earlystopping_cb]
                       )
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10,15))
    
    predictedTestLabel = []
    right=0
    wrong=0
    ys=[]
    for i,j in zip(X_test,y_test):
        y = model.predict(np.array(i).reshape(-1,len(X_train[0])))
        predictedTestLabel.append(y.argmax())
        '''
        We find out which class has the maximum value
        '''
        if j == y.argmax():
            right = right + 1
        else:
            wrong = wrong + 1
        ys.append(y.argmax())
    
    print("Errors: " + str(wrong), " Correct :" + str(right))
    
    print("Testing Accuracy: " + str(right/(right+wrong)*100))
    
    accuracies.append(str(right/(right+wrong)*100))













#Functions calls are done from code below this line



gsc_features_concat,gsc_features_sub=preprocess_gsc(2000)
human_features_concat,human_features_sub=preprocess_human_observed(791)

datasets=[]
datasets.append(human_features_concat)
datasets.append(human_features_sub)
datasets.append(gsc_features_concat)
datasets.append(gsc_features_sub)


datasetNames=["Human_Observed_Concat","Human_Observed_Sub","GSC_Concat","GSC_Sub"]

lo_count=0
lrs=[0.005,0.005,0.005,0.005]
i=0
for dataset in datasets:
    print(datasetNames[i])
    logistic_regression(dataset,lrs[i])
    i+=1
    lo_count+=1



ne_count=0
accuracies=[]
lrs=[0.005,0.005,0.001,0.005]
i=0
for dataset in datasets:
    print(datasetNames[i])
    neural_network(dataset)
    i+=1
    ne_count+=1

li_count=0
i=0
for dataset in datasets:
    print(datasetNames[i])
    linear_regression(dataset)
    i+=1
    li_count+=1



print ('UBITname      = vharisha')
print ('Person Number = 50291399')



