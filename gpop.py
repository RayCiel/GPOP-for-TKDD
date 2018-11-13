def gpop(pnew,S,A,l,k,t1,t2,R,lda,varargin):
    #params = inputParser;
    #params.addParameter('verbose', 'iter', @(x) ismember(x,{'iter','final','off'}));
    #params.addParameter('normdata',false,@islogical); % whether to normalize data in factoziation
    #params.parse(varargin{:});

    C = find_node_groups(S, A, l)
    n = size(S, 3)
    print("Finish finding node group..")

    (Pg, X, Xnorm, psize_t1) = top_k_similar(pnew, k, S, C, t1)
    (Pg_1, X_1, Xnorm_1, psize_1) = top_k_similar(Pnew, k, S, np.ones(n, 1), t1)
    Pa = [Pg_1, X_1, Xnorm_1, psize_1]
    print("Finish finding top-k similar processes..")

    traindata = create_hierarchical_data(Xnorm, pnew, Pg, Pa, t1, t2)
    Fac = factorize_tensors(traindata,R,lda,'verbose',params.Results.verbose,'normdata',params.Results.normdata) #?????

    pred = tensor(ktensor({Fac.K,Fac.J[t1:t2-1,:],Fac.F}))*psize_t1(pnew);#???
    pred = float(pred[0,:,:])
    gtruth = float(X[pnew-1,t1:t2-1,:])
    error = np.linalg.norm(pred-gtruth, ord = 'fro');
    print("Finish prediction..")
