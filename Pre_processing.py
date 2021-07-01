from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

def convert_no_num(X):
    if X=='one':
        return 1
    if X=='two':
        return 2
    if X=='three':
        return 3
    if X=='four':
        return 4
    if X=='five':
        return 5
    if X=='six':
        return 6
    if X=='seven':
        return 7
    if X=='eight':
        return 8
    if X=='nine':
        return 9
    if X=='ten':
        return 10
    if X=='eleven':
        return 11
    if X=='twelve':
        return 12

def featureScaling(X,a,b):
    Normalized_X=np.zeros((X.shape[0],X.shape[1]));
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a;
    return Normalized_X