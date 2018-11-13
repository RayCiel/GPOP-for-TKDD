def create_hierachical_data(X, pid, topkgroup, topksum, t1, t2):
    kgroup = len(topkgroup)
    ksum = len(topksum)
    l = size(X,3)

    if len(X[X[topkgroup,1:t2,:] != 0]) != 0:
        T = sptensor([kgroup, t2, l])
    else:
        T = X[topkgroup,1:t2,:].reshape([kgroup, t2, l])


    if len(X[X[topksum,1:t2,:] != 0]) != 0:
        Y = sptensor([ksum, t2, 1])
    else:
        Y = sptensor([ksum, t2, 1])
        Y[:,:,1] = collapse(X[topksum,1:t2,:].reshape([kgroup, t2, l]))

    if len(X[X[pid,t1:t2,:] != 0]) != 0:
        Z = sptensor([1, t1, l])
    else:
        Z = X[pid,1:t1,:].reshape([1, t1, l])

    if len(X[X[pid,1:t2,:] != 0]) != 0:
        Q = sptensor([1, t1, 1])
    else:
        Q = tensor(sptensor([1,t1,1]))
        if t1 == 1:
            Q(1,:,1) = collapse(X[pid,1:t1,:])
        else:
            tmp = tensor(collapse(X[pid,1:t1,:],2))
            Q(1,:,1) = tmp
    traindata.T = T;
    traindata.Y = Y;
    traindata.Z = Z;
    traindata.Q = Q;
    return traindata
