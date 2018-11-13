function [topk,X,Xnorm,psize_t1] = top_k_similar(pid,k,S,C,t1)
% find top-k similar processes for a given process pid 
%   S is a 3-dimensional sptensor: process x time x node
%   C is a vector containing the group membership of nodes
%   pid is the index of the process in S to be used as the query
%   t1 is the querying time, used for normalizing data
%   k is the number of topk items to return
%
%   topk = top_k_similar(pid,S,C,t1) find the topk similar processes to pid,
%       with regards to the groups in C. topk is a vector of indices in S.
%
%   [topk,X,Xnorm,psize_t1] = top_k_similar(pid,k,S,C,t1) also returns the 
%       group-based network state tensor X, its normalized version Xnorm,
%       and the processes' sizes psize_t1 at time t1

%% collapse the data in S into group-based network state tensor X
X = collapse_group(S,C);

%% normalize the data at time t1 
[Xnorm,psize_t1] = normalize_tensor(X,t1);

%% compute distance from pid to all other processes
% convert Xnorm to matrix
m = double(tenmat(Xnorm(:,1:t1,:),1));
% compute distance between pid and other processes
d = pdist2(m,m(pid,:));

%% compute the outlierness score of all processes
% normalize by last timestamp
[Xnorm2,psize_t2] = normalize_tensor(X,size(X,2));
% convert Xnorm to matrix
mfull = double(tenmat(Xnorm2,1));
dfull = pdist(mfull);
dmean=mean(dfull);

%% update distance with outlierness score
d = dmean.*d;

%% get the indices with smallest values in d
[vs,idx] = sort(d);
topk = idx(1:k+1);
% remove pid from topk
topk = setdiff(topk,pid);
topk = topk(1:k);