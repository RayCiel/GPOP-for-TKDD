def top_k_similar(pid, k, S, C, t1):
    X = collapse_group(S, C)
    (Xnorm, psize_t1) = normalize_tensor(X, t1)
    m = float(Xnorm[:,0:t1-1,:])#??????
    dd = np.vstack([m, m(pid,:)])
    d = pdist(dd)

    (Xnorm2, psize_t2) = normalize_tensor(X, size(X, 2))
    mfull = float(Xnorm2)#???????????
    dfull = pdist(mfull)
    dmean = np.mean(dfull)

    d = multiply(dmean, d)

    idx = np.argsort(d)
    topk = idx[0:k]
    topk = setdiff1d(topk, pid)
    topk = topk[0:k-1]
    return topk,X,Xnorm,psize_t1
