def find_node_groups(S, A, l):
    m = float(size(S, 1))
    q = folat(size(S, 2))
    n = float(size(S, 3))
    numVertices = m * q + n
    subs = np.argwhere(S!=0)
    vals = S[S!=0]
    subs[:, 0] = (subs[:,0] - 1) * q + subs[:, 1] + n
    vals_A = A[A != 0]
    rows_A = np.argwhere(A != 0)[:, 0]
    cols_A = np.argwhere(A != 0)[:, 1]
    I = [subs[:,2];subs[:,0];rows_A;[0:n-1]']#???分号，n*1
    J = [subs[:,0];subs[:,0];cols_A;[0:n-1]']#????分号，n*1
    K = [vals; vals; vals_A; np.ones(n,1)]#????n*1
    gstar = np.zeros(numVertices, numVertices)
    for i in range(len(I)):
        gstar[I[i],J[i]] = K[i]

    options = struct()#?????
    options.wgtflag = true; #??????/
    options.adjwgt = true;  #??????///
    clusall = metismex('PartGraphKway',gstar,l,options)+1;

    clusV = clusall[0:n-1];
    return clusV
