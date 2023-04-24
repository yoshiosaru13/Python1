from ctypes import string_at
from tkinter import Label
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

from sklearn.preprocessing import LabelEncoder
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

import numpy as np
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True).split(X_train, y_train)
scores = []

#for k, (train, test) in enumerate(kfold):
    #pipe_lr.fit(X_train[train], y_train[train])
    #score = pipe_lr.score(X_train[train], y_train[train])
    #scores.append(score)
    #print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
    
#print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr,
                         X = X_train, y = y_train,
                         cv=10, n_jobs=-1)


print('CV accuracy: scores: %s' % scores)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

