import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import auxilary2

REMOVE_OUTLIER = False


X_train = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task2\X_train.csv")
y_train = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task2\y_train.csv")

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

# Oversampling
#X_train, y_train = auxilary2.OverSample(X_train, y_train, 0, 0, 0.001)
print("Shape of X_train after oversampling: ", X_train.shape)
print("Shape of y_train after oversampling: ", y_train.shape)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


y_train = y_train.ravel()
kFold = 10
#qualityMeasures = ['gini', 'entropy']
treeValues = [1,2,3,4,5,6,7,8,9,10]
bestScore = np.NINF
bestNumber = 0
print("Computation starts")
#treeValues = [1]

for number in treeValues:
    print("Measure: ", number)
    kf = KFold(n_splits = kFold, shuffle = True, random_state=7)
    scores = []
    for train_index, test_index in kf.split(X_train,y_train):

        X_traincv, Y_traincv = X_train[train_index],y_train[train_index]
        X_testcv, Y_testcv = X_train[test_index], y_train[test_index]

        rfc = RandomForestClassifier(n_estimators=number, criterion='entropy', class_weight='balanced', random_state=64)
        rfc.fit(X_traincv, Y_traincv)
        y_pred = rfc.predict(X_testcv)
        score = balanced_accuracy_score(Y_testcv, y_pred)
        scores.append(score)

        
    averagedScore = (sum(scores)/kFold)
    if averagedScore > bestScore:
        bestNumber = number
        bestScore = averagedScore


print("best score is ", bestScore)
print("best measure is ", bestNumber)


rfc = RandomForestClassifier(n_estimators=bestNumber, criterion='entropy', class_weight='balanced', random_state=64)
rfc.fit(X_train, y_train)



X_test = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task2\X_test.csv")
del X_test['id']
X_test = X_test.values
X_test = scaler.fit_transform(X_test)
y_pred_test = rfc.predict(X_test)
auxilary2.createSubmissionFiles(y_pred_test)
