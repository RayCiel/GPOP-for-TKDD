function traindata = create_hierarchical_data(X, pid, topkgroup, topksum, t1, t2)
% create training data T,Y,Z,Q for prediction, given the group-based network tensor
% X, and the top-k results for group & aggregated states
% only keep data from time 1 to t2
%

kgroup = length(topkgroup);
ksum = length(topksum);
l = size(X,3); % number of groups

% create T
if nnz(X(topkgroup,1:t2,:)) == 0
    T = sptensor([kgroup,t2,l]);
else
    T = reshape(X(topkgroup,1:t2,:),[kgroup,t2,l]);
end

% create Y
if nnz(X(topksum,1:t2,:)) == 0
    Y = sptensor([ksum,t2,1]);
else    
    Y = sptensor([ksum,t2,1]);
    Y(:,:,1) = collapse(reshape(X(topksum,1:t2,:),[ksum,t2,l]),3);
end

% create Z
if nnz(X(pid,t1:t2,:)) == 0
    Z = sptensor([1,t1,l]);
else
    Z = reshape(X(pid,1:t1,:),[1,t1,l]);
end

% create Y
if nnz(X(pid,t1:t2,:)) == 0 
    Q = sptensor([1,t1,1]);
else
    Q = tensor(sptensor([1,t1,1]));
    if t1 == 1
        Q(1,:,1) = collapse(X(pid,1:t1,:));
    else
        tmp = tensor(collapse(X(pid,1:t1,:),2));
        Q(1,:,1) = tmp;
    end    
end

traindata.T = T;
traindata.Y = Y;
traindata.Z = Z;
traindata.Q = Q;