def collapse_group(S,C,l):
    if l.empty() or l == None:
        l = max(C)
    subs = np.argwhere(S!=0)
    vals = S[S!=0]
    subs[:,2] = C[subs[:,2]]
    X = sptensor(float(subs), float(vals), float([size(S, 1), size(S, 2), l]))
    return X
