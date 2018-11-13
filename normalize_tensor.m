function [Xnorm,psize] = normalize_tensor(X,t1)
% Normalize the data in the tensor by the size of each process at time t1
%   X is a 3-dimensional sptensor: process x time x group
%   t1 is a time at which we normalize X 
%
%   [Xnorm,psize] = normalize_tensor(X,t1)
%       Xnorm: the normalized version of X at time t1
%       psize: the size of the processes at time t1
%   
%% get the size of each process at time t1
psize = double(collapse(X(:,t1,:),2));

%% normalize X
[subs,vals] = find(X);
vals = vals./psize(subs(:,1));

Xnorm=sptensor(subs,vals,size(X));
