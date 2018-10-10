#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 06:40:50 2018

@author: varshath
"""

import csv
import numpy as np
import math
from sklearn.cluster import KMeans

from sklearn.feature_selection import VarianceThreshold

def getInputFeaturesFromCSV(file):
    data=[]
    with open(file) as csvfile:
        readCSV=csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            rowData=[]
            for col in row:
                rowData.append(float(col))
            data.append(rowData)
    data=np.delete(data,[5,6,7,8,9],axis=1)
    return data

def getTargetVectorsFromCSV(file):
    data=[]
    with open(file) as csvfile:
        readCSV=csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            data.append(int(row[0]))
    return data

def splitData(data,offset,percent):
    dataSize=int(len(data)*percent/100)
    dataSplit=data[offset:offset+dataSize+1]
    offset+=dataSize
    return dataSplit,offset

def getBigSigma(data):
    selector = VarianceThreshold()
    selector.fit_transform(data)
    varianceVector=selector.variances_
    bigSigma=np.zeros((len(data[0]),len(data[0])))
    for i in range(len(data[0])):
        bigSigma[i][i]=varianceVector[i]
    bigSigma=np.dot(bigSigma,1000)
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

def getDesignMatrix(data,mean,bigSigma):
    bigSigmaInverse=np.linalg.inv(bigSigma)
    designMatrix=np.zeros((len(data),len(mean)))
    for i in range(len(data)):
        for j in range(len(mean)):
            dataRow=trainData[i]
            m=mean[j]
            designMatrix[i][j]=findScalarVal(dataRow,m,bigSigmaInverse)
    return designMatrix

def getWeights(designMatrix,regularizationTerm):
    designMatrixTranspose=np.transpose(designMatrix)
    identityMatrix=np.identity(len(designMatrix[0]))
    identityMatrix*=regularizationTerm
    X=np.dot(designMatrixTranspose,designMatrix)
    X=np.add(identityMatrix,X)
    X=np.linalg.inv(X)
    X=np.dot(X,designMatrixTranspose)
    X=np.dot(X,trainTargetVectors)
    return X

def erms(predictedValues,actualValues):
    totalSum=0
    correctPrediction=0
    for i in range(len(predictedValues)):
        diff=actualValues[i]-predictedValues[i]
        diff*=diff
        totalSum+=diff
        if (int(predictedValues[i])==actualValues[i]):
            correctPrediction+=1
    percent=correctPrediction*100/len(actualValues)
    rms=math.sqrt(totalSum/len(actualValues))
    return percent,rms
    
    

inputFeatures=getInputFeaturesFromCSV("Querylevelnorm_X.csv")
targetVector=getTargetVectorsFromCSV("Querylevelnorm_t.csv")

offset=0
trainData,offset=splitData(inputFeatures,offset,80)
validationData,offset=splitData(inputFeatures,offset,10)
testData=inputFeatures[offset:]
offset=0
trainTargetVectors,offset=splitData(targetVector,offset,80)
validationTargetVector,offset=splitData(targetVector,offset,10)
testTargetVector=targetVector[offset:]
bigSigma=getBigSigma(trainData)

#Closed Form Solution

mean=getMean(trainData,3)
trainDesignMatrix=getDesignMatrix(trainData,mean,bigSigma)

regularizationTerm=0.9
weights=getWeights(trainDesignMatrix,regularizationTerm)

predictedTrainingVals=np.dot(weights,np.transpose(trainDesignMatrix))
trainingAccuracy,train_erms=erms(predictedTrainingVals,trainTargetVectors)


validationDesignMatrix=getDesignMatrix(validationData,mean,bigSigma)
validationPredictedVals=np.dot(weights,np.transpose(validationDesignMatrix))
validationAccuracy,valid_erms=erms(validationPredictedVals,validationTargetVector)



testDesignMatrix=getDesignMatrix(testData,mean,bigSigma)
testPredictedVals=np.dot(weights,np.transpose(testDesignMatrix))
testAccuracy,test_erms=erms(testPredictedVals,testTargetVector)

print ('UBITname      = vharisha')
print ('Person Number = 50291399')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = 10 \nLambda = 0.9")
print ("E_rms Training   = " + str(train_erms))
print ("E_rms Validation = " + str(valid_erms))
print ("E_rms Testing    = " + str(test_erms))





#SGD solution



currentWeights=np.transpose(weights)
currentWeights*220
learningRate=0.01
w=[]
trainingAccuracies=[]
training_erms=[]
validationAccuracies=[]
validation_erms=[]
testing_erms=[]
testingAccuracies=[]
for i in range(400):  
    deltaED= -np.dot((trainTargetVectors[i] - np.dot(np.transpose(currentWeights),trainDesignMatrix[i])),trainDesignMatrix[i])
    #print(deltaED)
    deltaEW=currentWeights*regularizationTerm
    deltaE=np.add(deltaED,deltaEW)
    deltaW=-(deltaE*(learningRate))
    newWeights=currentWeights+deltaW
    w.append(newWeights)
    currentWeights=newWeights
    
    
    predictedTrainingVals=np.dot(newWeights,np.transpose(trainDesignMatrix))
    trainingAccuracy,train_erms=erms(predictedTrainingVals,trainTargetVectors)
    
    trainingAccuracies.append(trainingAccuracy) 
    training_erms.append(train_erms)
    
    
    validationPredictedVals=np.dot(newWeights,np.transpose(validationDesignMatrix))
    validationAccuracy,valid_erms=erms(validationPredictedVals,validationTargetVector)
    validationAccuracies.append(validationAccuracy)
    validation_erms.append(valid_erms)
    
    
    testPredictedVals=np.dot(newWeights,np.transpose(testDesignMatrix))
    testAccuracy,test_erms=erms(testPredictedVals,testTargetVector)
    testingAccuracies.append(testAccuracy)    
    testing_erms.append(test_erms)


        

print ('----------Gradient Descent Solution--------------------')
print ("E_rms Training   = " + str(np.around(min(training_erms),5)))
print ("E_rms Validation = " + str(np.around(min(validation_erms),5)))
print ("E_rms Testing    = " + str(np.around(min(testing_erms),5)))