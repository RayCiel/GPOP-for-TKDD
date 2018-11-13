def factorize_tensors(traindata,R,lda,varargin):

    Z = traindata.Z
    Y = traindata.Y
    T = traindata.T
    Q = traindata.Q
    numClus = size(T,3)
    numTrainPgroup = size(T,1)
    numTrainPsum = size(Y,1)
    t1 = size(Z,2)
    t2 = size(T,2)
    trainT = 0:t1-1
    testT = t1:t2-1
    numTrainT = t1
    numTestT = t2 - t1

    for t = t1 : t2-1:
        T[:,t-1,:] = T[:,t-1,:] - T[:,t1-1,:]
        Y[:,t-1,:] = Y[:,t-1,:] - Y[:,t1-1,:]

    if (len(T[T != 0]) != 0):
        A = sptensor([numClus, numTrainPgroup, numTrainT])
        B = sptensor([numClus, numTrainPgroup, numTestT])
    else:
        subs = np.argwhere(T!=0)
        vals = T[T!=0]
        idx = (subs[:,1] <= t1)
        if (len(idx[idx > 0]) > 0):
            A = sptensor(subs[idx, [2,0,1]], vals[idx], [numClus, numTrainPgroup, numTrainT])
        else:
            A = sptensor([numClus, numTrainPgroup, numTrainT])
        idx = (subs[:,1] > t1)
        if (len(idx[idx > 0]) > 0):
            subs[:,1] = subs[:,1] - t1
            B = sptensor(subs[idx,[3,1,2]], vals[idx], [numClus, numTrainPgroup, numTrainT])
        else:
            B = sptensor([numClus, numTrainPgroup, numTestT])

    if (len(Y[Y != 0]) != 0):
        D = sptensor([1, numTrainPgroup, numTrainT])
        E = sptensor([1, numTrainPgroup, numTestT])
    else:
        subs = np.argwhere(Y != 0)
        vals = Y[Y != 0]
        idx = (subs[:,1] <= t1)
