import numpy as np
def sptensor(subs, vals):
    A = size(subs, 1)
    list = []
    for i in range(A):
        list.append(max(subs[i,:]))
    Mat = np.zeros(list)
    B = size(subs, 2)
    for i in range(B):
        Mat[tuple(subs[:,i])] = vals[i]
    return Mat

def sptensor(subs, vals, list):
    A = size(subs, 1)
    B = size(subs, 2)
    Mat = np.zeros(list)
    for i in range(B):
        Mat[tuple(subs[:,i])] = vals[i]
    return Mat

def sptensor(list):
    Mat = np.zeros(list)
    return Mat
