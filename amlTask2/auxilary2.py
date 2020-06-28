import numpy as np
import numpy.linalg as LA
from scipy import linalg as LA2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
seedValue = 1
np.random.seed(seedValue)
import random
random.seed(seedValue)
from sklearn.ensemble import IsolationForest

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_classif

import tensorflow as tf
import keras
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import BatchNormalization	
from keras.utils import np_utils
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers



DEBUG = False



def UnderSample(X_train, y_train, numberOfSamplesToRemoveFromClass1):

    # Get indices of rows with class equals 1 shuffle them and select 3000 to remove    
    indicesOfClass1 = np.where(y_train == 1)[0]
    np.random.shuffle(indicesOfClass1)
    indicesOfClass1 = indicesOfClass1[:numberOfSamplesToRemoveFromClass1]

    # Remove these indices, from X_train and y_train
    X_train = np.delete(X_train, indicesOfClass1, axis=0)
    y_train = np.delete(y_train, indicesOfClass1, axis=0)

    print("Number Of Samples with Class 0: ", np.count_nonzero(y_train == 0))
    print("Number Of Samples with Class 1: ", np.count_nonzero(y_train == 1))
    print("Number Of Samples with Class 2: ", np.count_nonzero(y_train == 2))

    return X_train, y_train

def OverSample(X_train, y_train, numberOfSamplesToIntroduce, mean, variance):
    indicesOfClass0 = np.where(y_train == 0)[0]
    indicesOfClass2 = np.where(y_train == 2)[0]

    class0Train = X_train[indicesOfClass0, :]
    class2Train = X_train[indicesOfClass2, :]


    print(class0Train.shape)
    print(class2Train.shape)
    print(y_train.shape)

    for i  in range(5):
        if i == 0:
            class0OS = class0Train + np.random.normal(mean, variance, class0Train.shape)
            class2OS = class2Train + np.random.normal(mean, variance, class2Train.shape)
        else:
            class0OS = np.concatenate((class0OS, class0Train + np.random.normal(mean, variance, class0Train.shape)), axis = 0)
            class2OS = np.concatenate((class2OS, class2Train + np.random.normal(mean, variance, class2Train.shape)), axis = 0)


    yValues = np.zeros((3000,1))
    yValues = np.concatenate((yValues, 2*np.ones((3000,1))))
    y_train = np.concatenate((y_train, yValues), axis = 0)

    print(y_train)
    print("adasdas", y_train.shape)
    print(class0OS.shape)
    print(class2OS.shape)     
    print(X_train.shape)
    
    X_train = np.concatenate((X_train, class0OS, class2OS), axis = 0)
    print(X_train.shape)

    return X_train, y_train


def getModel(inputDimension):
    model = Sequential()

    model.add(Dense(2048, input_dim=inputDimension))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(2048, use_bias = True, kernel_initializer='random_uniform', bias_initializer = 'zeros' ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # model.add(Dense(2048, use_bias = True, kernel_initializer='random_uniform', bias_initializer = 'zeros' ))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    
    # model.add(Dense(1024, use_bias = True, kernel_initializer='random_uniform', bias_initializer = 'zeros' ))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    
    # model.add(Dense(1024, use_bias = True, kernel_initializer='random_uniform', bias_initializer = 'zeros' ))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(3, activation='softmax'))

    return model


def balancedmcaccuracy(y_true, y_pred):
    class_accuracy = 0
    for i in range(3):
        class_true = K.argmax(y_true, axis = -1)
        class_pred = K.argmax(y_pred, axis = -1)
        accuracy_mask = K.cast(K.equal(class_true, i), 'int32')
        class_acc_tensor = K.cast(K.equal(class_true, class_pred), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        class_accuracy += class_acc
    return class_accuracy / 3

    

def plotSFeatureScores(X_train, y_train, score_function):
    inputDim = 200
    featureSelection = SelectKBest(score_function, k = inputDim)
    featureSelection.fit(X_train, y_train)
    scores = featureSelection.scores_
    scores = np.sort(scores)[::-1]
    axis = np.arange(1,X_train.shape[1]+1)
    scores = np.nan_to_num(scores, nan = 0)
    total = np.sum(scores)
    #print("total: ", total)
    plt.plot(axis, scores, 'ro',marker = '+', markersize = 0.4)
    plt.plot(axis, scores/total, 'ro', marker = '+', markersize = 0.4)
    plt.show()





def zeroMean(data):
    colMean = np.nanmean(data, axis = 0)
    if DEBUG:
        print("The mean of features: ", colMean)
    data -= colMean
    data = np.nan_to_num(data, nan = 0)
    return data




def OutlierDetectionIsolationForest(data, labels, percentageOutlier):
    
    clf = IsolationForest( behaviour = 'new', max_samples=0.99, random_state = 1, contamination= percentageOutlier)
    preds = clf.fit_predict(data)

    indicesToRemove = np.argwhere(preds == -1)
    numberOfOutliers = np.count_nonzero(preds == -1)
    print("Number Of Outliers:", numberOfOutliers)
    data = np.delete(data, indicesToRemove, axis = 0)
    labels = np.delete(labels, indicesToRemove)

    return data, labels



def createSubmissionFiles(y_predictions):
    output = pd.DataFrame()
    output.insert(0, 'y', y_predictions)
    A = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task2\X_test.csv")
    output.index = A.index
    output.index.names = ['id']
    output.to_csv("output")
    print("Submission files are succesfully created")












































"""" KNN APPROACH WITH CROSVALIDATION ON NUMBER OF NEIGHBORS """
# y_train = y_train.ravel()
# kFold = 10
# kValues = np.arange(2, 10, 1)
# bestScore = np.NINF
# bestK = 0

# print("Computation starts")

# for kvalue in kValues:
#     print("Kvalue: ", kvalue)
#     kf = KFold(n_splits = kFold, shuffle = False)
#     scores = []
#     for train_index, test_index in kf.split(X_train,y_train):

#         X_traincv, Y_train = X_train[train_index],y_train[train_index]
#         X_test, Y_test = X_train[test_index], y_train[test_index]

#         kNN = KNeighborsClassifier(n_neighbors=kvalue)
#         kNN.fit(X_train, y_train)
#         y_pred = kNN.predict(X_test)
#         score = balanced_accuracy_score(Y_test, y_pred)
#         scores.append(score)

        
#     averagedScore = (sum(scores)/kFold)
#     if averagedScore > bestScore:
#         bestK = kvalue
#         bestScore = averagedScore


# print("best score is ", bestScore)
# print("best kvalue is ", bestK)



# kNN = KNeighborsClassifier(n_neighbors=bestK)
# kNN.fit(X_train, y_train)

# X_test = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task2\X_test.csv")
# del X_test['id']
# X_test = X_test.values
# X_test = scaler.fit_transform(X_test)
# y_pred_test = kNN.predict(X_test)
# auxilary2.createSubmissionFiles(y_pred_test)



