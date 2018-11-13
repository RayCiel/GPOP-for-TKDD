function [pred,gtruth,error,C,X] = gpop(pnew,S,A,l,k,t1,t2,R,lda,varargin)
% make prediction for process pnew, using all other processes in S
%   S is a 3-dimensional sptensor: process x time x node
%   A is a symmetric sparse matrix: node x node
%   pnew is the index of the process in S to be predicted
%   t1 is the querying time, used for normalizing data
%   k is the number of topk 
%   t1: maximum observed timestamp
%   R : number of latent dimensions
%   lda: regularization lambda for tensor decomposition
%   l: number of groups
%
%
%   'verbose' - whether to print each iteration of the gradient descent
%   algorithm, can be either one three following strings:
%       'final' : only print the final iteration
%       'off' : do not print
%       'iter' : print each iter
%
%   'normdata': TRUE/FALSE(default) whether to normalize data in
%   factorization, which may make the learning faster
%
params = inputParser;
params.addParameter('verbose', 'iter', @(x) ismember(x,{'iter','final','off'}));
params.addParameter('normdata',false,@islogical); % whether to normalize data in factoziation
params.parse(varargin{:});

%% find user groups
C = find_node_groups(S,A,l);
n = size(S,3);
fprintf('Finish finding node group.\n');

%% find top_k 
% group-based topk
[Pg,X,Xnorm,psize_t1] = top_k_similar(pnew,k,S,C,t1);
% aggregated topk
Pa = top_k_similar(pnew,k,S,ones(n,1),t1);
fprintf('Finish finding top-k similar processes.\n');

%% prepare data for prediction
traindata = create_hierarchical_data(Xnorm,pnew,Pg,Pa,t1,t2);

%% factorize tensors
Fac = factorize_tensors(traindata,R,lda,'verbose',params.Results.verbose,...
    'normdata',params.Results.normdata);

%% make prediction
pred = tensor(ktensor({Fac.K,Fac.J(t1+1:t2,:),Fac.F}))*psize_t1(pnew);
pred = double(pred(1,:,:));
gtruth = double(X(pnew,t1+1:t2,:));
error = norm(pred-gtruth,'fro');
fprintf('Finish prediction.\n');