import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA, PLSSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
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

y_train = y_train.ravel()

# Feature Selection !
inputDim = 500
# featureSelection = SelectKBest(f_classif, k = inputDim)
# X_train = featureSelection.fit_transform(X_train, y_train)
# #scores = featureSelection.scores_
# print("Shape after feature selection: ", X_train.shape)

#pca = PCA(n_components=inputDim, svd_solver='auto')
#X_train = pca.fit_transform(X_train)


# # lda = LDA(n_components=inputDim, solver='svd', priors=[600/4800, 3600/4800, 600/4800])
# # X_train = lda.fit_transform(X_train, y_train)

# # Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

parameters = { #'kernel': ('linear', 'rbf', 'sigmoid', 'poly'),
                'C': [0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1],
                'coef0': [1.0, 2.0, 3, 4, 5, 6, 7, 8, 9,10]
                }

svc = SVC(kernel='poly', gamma = 'scale', class_weight='balanced', random_state=0, max_iter=40000,
            decision_function_shape= 'ovo', degree=3)

scoreFunction = make_scorer(balanced_accuracy_score, greater_is_better=True)

clf = GridSearchCV(svc, parameters, cv=10, verbose=2, scoring= scoreFunction, n_jobs=16)
clf.fit(X_train, y_train)


print("Best score of best on validation set: ", clf.best_score_)
print("Best Parameters: ", clf.best_params_)


X_test = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task2\X_test.csv")
del X_test['id']
X_test = X_test.values
#X_test = featureSelection.transform(X_test)
#X_test = pca.transform(X_test)
X_test = scaler.fit_transform(X_test)
y_pred_test = clf.predict(X_test)
auxilary2.createSubmissionFiles(y_pred_test)
