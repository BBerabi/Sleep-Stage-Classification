import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import auxilary2

REMOVE_OUTLIER = True


X_train = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task2\X_train.csv")
y_train = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task2\y_train.csv")

X_train = X_train.drop(columns='id', axis=1)
y_train = y_train.drop(columns='id', axis=1)
y_train = y_train.values
X_train = X_train.values

# No missing values in the data
#print("Number of missing values in train is ", np.sum(np.isnan(X_train)))

# Set true to remove outliers, there are 31 of them. Not sure if it makes sense for this calssficiation task. Therefore False for now.
if REMOVE_OUTLIER:
    X_train, y_train = auxilary2.OutlierDetectionIsolationForest(X_train, y_train, percentageOutlier = 'auto')
    print("Shape after outlier detection: ", X_train.shape)

# Uncomment to plot feature scores, there is no knie in the graph so probably no need for feature selection
# auxilary2.plotSFeatureScores(X_train, y_train, f_classif)


# Feature Selection !
#inputDim = 100
#featureSelection = SelectKBest(f_classif, k = inputDim)
#X_train = featureSelection.fit_transform(X_train, y_train)
#scores = featureSelection.scores_
print("Shape after feature selection: ", X_train.shape)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

y_train = y_train.ravel()
kFold = 3
kernel = 'linear'
gamma = 'scale'
decision_func = 'ovo'

cValues = np.arange(0.6, 0.65, 0.1)
#cValues = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
cValues = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
#cValues = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#cValues = [1]
bestScore = np.NINF
bestScoreTrain = 0
bestC = 0

print("Computation starts")

for cvalue in cValues:
    splitCounter = 1
    print("Cvalue: ", cvalue)
    kf = StratifiedKFold(n_splits = kFold, shuffle = True, random_state=7)
    scores = []
    trainingScores = []

    for train_index, test_index in kf.split(X_train,y_train):
        X_traincv, Y_traincv = X_train[train_index], y_train[train_index]
        X_testcv, Y_testcv = X_train[test_index], y_train[test_index]

        # sm = SMOTE(random_state=0)
        # X_train_oversampled, y_train_oversampled = sm.fit_sample(X_traincv, Y_traincv)
        # X_train_oversampled = scaler.fit_transform(X_train_oversampled)

        svc = SVC(C = cvalue, kernel=kernel, gamma=gamma, 
                class_weight='balanced', decision_function_shape=decision_func,
                random_state=42, max_iter=10000)
        svc.fit(X_traincv, Y_traincv)
        
        y_pred = svc.predict(X_testcv) 
        score = balanced_accuracy_score(Y_testcv, y_pred)
        scores.append(score)
        
        y_prediction_trian = svc.predict(X_traincv)
        trainingScore = balanced_accuracy_score(Y_traincv, y_prediction_trian)
        trainingScores.append(trainingScore)
    

    averagedScore = sum(scores)/kFold
    averagedScoreTrain = sum(trainingScores)/kFold
    print("Training score: ", averagedScoreTrain)
    print("Test score: ", averagedScore)
    if averagedScore > bestScore:
        bestC = cvalue
        bestScore = averagedScore
        bestScoreTrain = averagedScoreTrain
        

print("best test score is ", bestScore)
print("best train score is ", bestScoreTrain)

print("best cvalue is ", bestC)

#sm = SMOTE(random_state=0)
#X_train_oversampled, y_train_oversampled = sm.fit_sample(X_train, y_train)

svc = SVC(C = bestC, kernel=kernel, gamma=gamma, 
                class_weight='balanced', decision_function_shape=decision_func,
                random_state=42, max_iter=10000)
svc.fit(X_train, y_train)

X_test = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task2\X_test.csv")
del X_test['id']
X_test = X_test.values
#X_test = featureSelection.transform(X_test)
#print("joke?")
#y_pred_train = bestModel.predict(X_train)
#totalTrainScore = balanced_accuracy_score(y_pred_train, y_train)
#print("Final train score is ", totalTrainScore)

X_test = scaler.fit_transform(X_test)
y_pred_test = svc.predict(X_test)
auxilary2.createSubmissionFiles(y_pred_test)
