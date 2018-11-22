#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:20:14 2018

@author: varshath
"""

import pickle
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from keras.utils import np_utils
from PIL import Image
import seaborn as sn
import os
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop,Adam
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
def softmax(z):
    return (np.exp(z-np.max(z)).T / np.sum(np.exp(z-np.max(z)),axis=1)).T

def get_mini_batches(x, y, batch_size):
    random_idxs = random.choice(len(y), len(y), replace=False)
    X_shuffled = x[random_idxs,:]
    y_shuffled = y[random_idxs]
    mini_batches = [(X_shuffled[i:i+batch_size,:], y_shuffled[i:i+batch_size]) for
                   i in range(0, len(y), batch_size)]
    return mini_batches


def predict(X,w):
    probabilities=get_class_probabilities(X,w)
    prediction=np.argmax(probabilities,axis=1)
    return prediction

def get_class_probabilities(X,w):
    return softmax(np.dot(X,w))

def convert_to_categorical(y):
    return np_utils.to_categorical(np.array(y),10)

def calculate_loss_and_gradient(X,y,w):
    y=convert_to_categorical(y)
    probabilities=get_class_probabilities(X,w)
    size=X.shape[0]
    loss = (-1 / size ) * np.sum(y * np.log(probabilities)) 
    gradient = (-1 / size ) * np.dot(X.T,(y - probabilities)) 
    return loss,gradient

def get_accuracy(X,y,w):
    predicted_values = predict(X,w)
    accuracy = sum(predicted_values == y)/(float(len(y)))
    return accuracy



def confusion_matrix(actual, predicted):
    cm=np.zeros([10,10])
    for i in range(len(predicted)):
        cm[actual[i]][predicted[i]]+=1
    return cm
    


def logistic_regression(training_data,testing_data,validation_data,epochs,leaning_rate,mini_batch_size,USPS_data,USPS_target):
    print()
    print("LOGISTIC REGRESSION")
    time.sleep(5)
    X_train,y_train,X_test,y_test=training_data[0],training_data[1],test_data[0],test_data[1]
    X_val,y_val=validation_data[0],validation_data[1]
    mini_batches = get_mini_batches(X_train,y_train,mini_batch_size)
    w = np.zeros([X_train.shape[1],len(np.unique(y_train))])
    training_accuracies=[]
    testing_accuracies=[]
    validation_accuracies=[]
    usps_accuracies=[]
    _loss=[]
    for i in range(epochs):
        for mini_batch in mini_batches:
            loss,gradient=calculate_loss_and_gradient(mini_batch[0],mini_batch[1],w)
            w-=(learning_rate * gradient)
        _loss.append(loss)
        training_accuracies.append(get_accuracy(X_train,y_train,w))
        testing_accuracies.append(get_accuracy(X_test,y_test,w))
        validation_accuracies.append(get_accuracy(X_val,y_val,w))
        usps_accuracies.append(get_accuracy(USPS_data,USPS_target,w))
    plt.plot(training_accuracies, label = 'Training Accuracy')
    plt.plot(validation_accuracies, label = 'Validation Accuracy')
    plt.plot(testing_accuracies,label = 'Test Accuracy')
    plt.plot(usps_accuracies,label="USPS Accuracy")
    
    plt.legend()
    plt.ylabel('Accuracy')
    plt.show()
    predicted_mnist=predict(X_test,w)
    cm=confusion_matrix(y_test,predicted_mnist)
    print("Confusion Matrix for Logistic Regression(MNIST)")
    print(np.round(cm,0))
    predicted_usps=predict(USPS_data,w)
    cm=confusion_matrix(USPS_target,predicted_usps)
    global h
    h=cm
    print("Confusion Matrix for Logistic Regression(USPS)")
    print(np.round(cm,2))
    return predicted_mnist,predicted_usps
        
        
            
def neural_network(training_data,testing_data,validation_data,num_epochs,learning_rate,mini_batch_size,USPS_data,USPS_target):
    print()
    print("DNN")
    time.sleep(5)
    X_train,y_train,X_test,y_test=training_data[0],training_data[1],test_data[0],test_data[1]
    X_val,y_val=validation_data[0],validation_data[1]
    X_train = np.concatenate((X_train,X_val),axis = 0)
    y_train = np.concatenate((y_train,y_val),axis = 0)
    input_size = len(X_train[0])
    drop_out = 0.5
    first_dense_layer_nodes  = 100
    second_dense_layer_nodes = 10
    
    validation_data_split = 0.20
    early_patience = 100
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

    
    model = Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('sigmoid'))
    model.add(Dropout(drop_out))
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation(tf.nn.softmax))
    opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    
    
    
    
    history = model.fit(X_train
                        , np_utils.to_categorical(np.array(y_train), 10)
                        , validation_split=validation_data_split
                        , epochs=num_epochs
                        , batch_size=mini_batch_size
                        , callbacks = [earlystopping_cb]
                       )
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10,15))
    plt.show()
    
    
    
    testing_data=[]
    testing_label=[]
    testing_data.append(X_test)
    testing_data.append(np.asmatrix(USPS_data))
    testing_label.append(y_test)
    testing_label.append(USPS_target)
    predicted_vals=[]
    dataset=["MNIST","USPS"]
    for t in range(len(testing_data)):
        processedTestData  = testing_data[t]
        processedTestLabel = testing_label[t]
        predictedTestLabel = []
        wrong   = 0
        right   = 0
        for i,j in zip(processedTestData,processedTestLabel):
            
            y = model.predict(np.array(i).reshape(-1,processedTestData.shape[1]))
            predictedTestLabel.append((y.argmax()))
            if j == (y.argmax()):
                right = right + 1
            else:
                wrong = wrong + 1
        print(dataset[t]+":")
        print("Errors: " + str(wrong), " Correct :" + str(right))
        
        print("Testing Accuracy: " + str(right/(right+wrong)*100))
        
        cm = confusion_matrix(predictedTestLabel, processedTestLabel)
        
        print("Confusion Matrix for Neural Network ("+dataset[t]+")")
        print(np.round(cm,2))
        predicted_vals.append(predictedTestLabel)
        
    
    return predicted_vals[0],predicted_vals[1]
    
def get_model_for_convolution(kernel_size):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu'))
    model.add(Flatten())
    softmax=Activation(tf.nn.softmax)
    model.add(Dense(10, activation=softmax))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    return model           

    
def convolutional_neural_network(training_data,test_data,validation_data,USPS_data,USPS_target):
    print()
    print("CNN")
    time.sleep(5)
    X_train,y_train,X_test,y_test=training_data[0],training_data[1],test_data[0],test_data[1]
    X_val,y_val=validation_data[0],validation_data[1]
    MNIST_accuracy=[]
    USPS_accuracy=[]
    kernels=[3]
    for kernel in kernels:
        model=get_model_for_convolution(kernel)
        model.fit(X_train.reshape(50000,28,28,1), to_categorical(y_train), validation_data=(X_val.reshape(10000,28,28,1), to_categorical(y_val)), epochs=3)
        score = model.evaluate(test_data[0].reshape(X_test.shape[0],28,28,1), to_categorical(y_test), verbose=0)
        MNIST_accuracy.append(score[1])
        score = model.evaluate((np.asarray(USPS_data)).reshape(len(USPS_data),28,28,1), to_categorical(USPS_target), verbose=0)
        USPS_accuracy.append(score[1])
        mnist_predicted = model.predict_classes(test_data[0].reshape(test_data[0].shape[0],28,28,1))
        usps_predicted = model.predict_classes((np.asarray(USPS_data)).reshape(len(USPS_data),28,28,1))
        print("Confusion Matrix for MNIST using CNN: ")
        print(confusion_matrix(mnist_predicted, test_data[1]))
        print("Confusion Matrix for USPS using CNN: ")
        print(confusion_matrix(usps_predicted,USPS_target))
    return mnist_predicted,usps_predicted


def obtain_accuracy(predicted,actual):
    return sum(predicted == actual)/(float(len(actual)))
    
def random_forest(training_data,test_data,validation_data,USPS_data,USPS_target):
    print()
    print("RANDOM FOREST")
    time.sleep(5)
    X_train,y_train,X_test,y_test=training_data[0],training_data[1],test_data[0],test_data[1]
    X_val,y_val=validation_data[0],validation_data[1]
    X_train = np.concatenate((X_train,X_val),axis = 0)
    y_train = np.concatenate((y_train,y_val),axis = 0)
    trees=[10*i for i in range(1,11)]
    #trees=[40]
    mnist_accuracies=[]
    usps_accuracies=[]
    for tree in trees:
        classifier=RandomForestClassifier(n_estimators=tree, verbose=1)
        classifier.fit(X_train,y_train)
        predicted_mnist=classifier.predict(X_test)
        mnist_accuracies.append(obtain_accuracy(predicted_mnist,y_test))
        predicted_usps=classifier.predict(USPS_data)
        usps_accuracies.append(obtain_accuracy(predicted_usps,USPS_target))
    print("Confusion Matrix for MNIST using Random Forest: ")
    print(confusion_matrix(predicted_mnist,y_test))
    print("Confusion Matrix for USPS using Random Forest: ")
    print(confusion_matrix(predicted_usps,USPS_target))
    plt.figure(1) 
    plt.scatter(range(trees[0],trees[-1]+10,10),mnist_accuracies,label = 'MNIST Accuracy',marker ="o")
    plt.scatter(range(trees[0],trees[-1]+10,10),usps_accuracies,label = 'USPS Accuracy', marker="^")
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('No:of Trees')
    plt.show()
    #plt.gcf().clear()
    return predicted_mnist,predicted_usps
       

def support_vector_machine(training_data,test_data,validation_data,USPS_data,USPS_target):
    print()
    print("SVM")
    time.sleep(5)
    X_train,y_train,X_test,y_test=training_data[0],training_data[1],test_data[0],test_data[1]
    X_val,y_val=validation_data[0],validation_data[1]
    X_train = np.concatenate((X_train,X_val),axis = 0)
    y_train = np.concatenate((y_train,y_val),axis = 0)
    mnist_accuracies=[]
    usps_accuracies=[]
    '''
    gammas=[0.01,0.05,0.09,1]
    for g in gammas:
        classifier = SVC(kernel='rbf', C=2, gamma = g)
        classifier.fit(X_train,y_train)
        predicted_mnist=classifier.predict(X_test)
        mnist_accuracy=obtain_accuracy(predicted_mnist,y_test)
        print ('Testing Accuracy MNIST: ', mnist_accuracy)
        predicted_usps =  classifier.predict(USPS_data)
        usps_accuracy=obtain_accuracy(predicted_usps,USPS_target)
        print ('testing Accuracy USPS: ', usps_accuracy)
        mnist_accuracies.append(mnist_accuracy)
        usps_accuracies.append(usps_accuracy)
    
    
    
    classifier = SVC(kernel='linear', C=2, gamma = 0.05)
    classifier.fit(X_train,y_train)
    predicted_mnist=classifier.predict(X_test)
    mnist_accuracy=obtain_accuracy(predicted,y_test)
    print ('Testing Accuracy MNIST: ', mnist_accuracy)
    predicted_usps =  classifier.predict(USPS_data)
    usps_accuracy=obtain_accuracy(predicted_usps,USPS_target)
    print ('testing Accuracy USPS: ', usps_accuracy)
    mnist_accuracies.append(mnist_accuracy)
    usps_accuracies.append(usps_accuracy)
    '''
    
    classifier = SVC(kernel='rbf', C=2,gamma=0.02)
    classifier.fit(X_train,y_train)
    predicted_mnist=classifier.predict(X_test)
    mnist_accuracy=obtain_accuracy(predicted_mnist,y_test)
    print ('Testing Accuracy MNIST(default gamma): ', mnist_accuracy)
    predicted_usps =  classifier.predict(USPS_data)
    usps_accuracy=obtain_accuracy(predicted_usps,USPS_target)
    print ('testing Accuracy USPS(defaul gamma): ', usps_accuracy)
    mnist_accuracies.append(mnist_accuracy)
    usps_accuracies.append(usps_accuracy)
    print("Confusion Matrix for MNIST using SVM: ")
    print(confusion_matrix(predicted_mnist,test_data[1]))
    print("Confusion Matrix for USPS using SVM: ")
    print(confusion_matrix(predicted_usps,USPS_target))
    
    return predicted_mnist,predicted_usps
    
    
def voting(predictions):
    voted_prediction=[]
    for i in range(len(predictions[0])):
        classes=np.zeros(10)
        for prediction in predictions:
            classes[prediction[i]]+=1
        voted_prediction.append(np.argmax(classes))
    return voted_prediction
    

filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()
USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []
for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)





epochs=100
learning_rate=0.01
mini_batch_size=1500

logistic_prediction_mnist,logistic_prediction_usps=logistic_regression(training_data,test_data,validation_data,epochs,learning_rate,mini_batch_size,USPSMat,USPSTar)

neural_net_prediction_mnist,neural_net_prediction_usps=neural_network(training_data,test_data,validation_data,epochs,learning_rate,mini_batch_size,USPSMat,USPSTar)

cnn_prediction_mnist,cnn_prediction_usps=convolutional_neural_network(training_data,test_data,validation_data,USPSMat,USPSTar)

random_forest_prediction_mnist,random_forest_prediction_usps=random_forest(training_data,test_data,validation_data,USPSMat,USPSTar)

svm_prediction_mnist,svm_prediction_usps=support_vector_machine(training_data,test_data,validation_data,USPSMat,USPSTar)



predictions=[]
predictions.append(np.asarray(logistic_prediction_mnist))
predictions.append(np.asarray(neural_net_prediction_mnist))
predictions.append(np.asarray(random_forest_prediction_mnist))
predictions.append(np.asarray(svm_prediction_mnist))
predictions.append(np.asarray(cnn_prediction_mnist))
voted_predictions=np.asarray(voting(predictions))
final_accuracy_mnist=obtain_accuracy(voted_predictions,test_data[1])



predictions=[]
predictions.append(np.asarray(logistic_prediction_usps))
predictions.append(np.asarray(neural_net_prediction_usps))
predictions.append(np.asarray(random_forest_prediction_usps))
predictions.append(np.asarray(svm_prediction_usps))
predictions.append(np.asarray(cnn_prediction_usps))
voted_predictions=np.asarray(voting(predictions))
final_accuracy_usps=obtain_accuracy(voted_predictions,USPSTar)

print("After Voting MNIST Accuracy = "+str(final_accuracy_mnist))
print("After Voting USPS Accuracy = "+str(final_accuracy_usps))


print()
print("UBIT NAME : VHARISHA")
print("PERSON NUMBER : 50291399")


