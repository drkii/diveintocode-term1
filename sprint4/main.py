import pandas as pd
import numpy as np
import svm
from sklearn import datasets
from sklearn import preprocessing
from sklearn import metrics
import os
from sklearn.model_selection import train_test_split
import math

if __name__=='__main__':

    files = os.listdir('./')
    print(os.getcwd())
    # iris = datasets.load_iris()
    # X = iris.data[:100]
    # y = iris.target[:100]
    # # label is 1 or -1
    # y = list(map(lambda l: 1 if l == 1 else -1, y))
    # # scaling
    # scaler = preprocessing.StandardScaler()
    # X = scaler.fit_transform(X)
    # clf = svm.SDGSVC()
    # clf.fit(X, y)
    # y_pred = clf.predict(X)
    # print("Confusion Matrix")
    # print(metrics.confusion_matrix(y, y_pred))
    # print(metrics.classification_report(y, y_pred))

    # breast cancer
    train_df = pd.read_csv('./wdbc.data', header=None)

    col_name = ['radius', 'texture', 'perimeter ', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', \
                'symmetry', 'fractal_dimension']

    col_name2 = ['_mean', '_ste', '_w_or_l']

    columns_dict = {0: 'id', 1: 'target'}

    count = 2
    for col2 in col_name2:
        for col in col_name:
            columns_dict[count] = col + col2
            count += 1
    # columns_dict
    train_df = train_df.rename(columns=columns_dict)
    y_df = train_df['target']
    y_list = list(map(lambda l: 1 if l == "M" else -1, y_df))

    X_df = train_df.drop(["id", "target"], axis=1)

    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X_df)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_list, test_size=0.2, random_state=0)

    clf = svm.SDGSVC(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Confusion Matrix")
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

    # print(len(clf.alpha.flatten()))
    # print(np.dot((clf.alpha * y_train).flatten(), X_train))
    W = np.dot((clf.alpha * np.array(y_train)[:, np.newaxis]).T, X_train)
    b = clf.bias
    w_len = math.sqrt((W * W).sum())

    distance = np.dot(X_train, W.T) + b
    distance /= w_len

    result_df = pd.DataFrame(clf.alpha)
    result_df['length'] = distance

    print(result_df)