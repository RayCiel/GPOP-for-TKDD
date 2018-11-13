def normalize_tensor(X, t1):
    psize = float(collapse(X[:,t1,:], 2))
    vals = S[X!=0]
    subs = np.argwhere(S!=0)
    Xnorm = sptensor(subs, vals, X.shape)
    return Xnorm, psize
