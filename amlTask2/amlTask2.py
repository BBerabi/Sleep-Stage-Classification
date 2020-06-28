import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils import class_weight

import auxilary2
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
from keras.utils import np_utils

REMOVE_OUTLIER = False

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_train = X_train.drop(columns='id', axis=1)
y_train = y_train.drop(columns='id', axis=1)
y_train = y_train.values
X_train = X_train.values

# No missing values in the data
print("Number of missing values in train is ", np.sum(np.isnan(X_train)))

# Set true to remove outliers, there are 31 of them. Not sure if it makes sense for this calssficiation task. Therefore False for now.
if REMOVE_OUTLIER:
    X_train, y_train = auxilary2.OutlierDetectionIsolationForest(X_train, y_train, percentageOutlier = 'auto')
    print("Shape after outlier detection: ", X_train.shape)


# Uncomment to plot feature scores, there is no knie in the graph so probably no need for feature selection
#auxilary2.plotSFeatureScores(X_train, y_train, f_classif)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42, shuffle = True)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
#X_train = np.nan_to_num(X_train, nan = 0)
#X_val = np.nan_to_num(X_val, nan = 0)


class_weights = class_weight.compute_class_weight('balanced', 
                                                    np.unique(y_train), 
                                                    y_train.ravel())
class_weights_dict = {}
for i in range(3):
    class_weights_dict[i] = class_weights[i]



monitorstring = 'val_balancedmcaccuracy'

model = auxilary2.getModel(X_train.shape[1])
es = EarlyStopping(monitor=monitorstring, mode='max', verbose=1, patience=500)
mc = ModelCheckpoint('best_model.h5', monitor=monitorstring, mode='max', verbose=1, save_best_only=True)
opt = keras.optimizers.Adam(lr = 0.0001)

y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=[auxilary2.balancedmcaccuracy])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 100, batch_size = 256, class_weight= class_weights_dict, callbacks=[es,mc])


bestModel = load_model('best_model.h5', custom_objects = {'balancedmcaccuracy': auxilary2.balancedmcaccuracy})
#bestModel = load_model('best_model.h5')


X_test = pd.read_csv("X_test.csv")
del X_test['id']
X_test = scaler.fit_transform(X_test)
y_predictions = bestModel.predict_classes(X_test)
y_predictions = np.argmax(to_categorical(y_predictions), axis = 1)
auxilary2.createSubmissionFiles(y_predictions)













