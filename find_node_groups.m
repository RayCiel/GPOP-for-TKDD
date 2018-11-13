function clusV = find_node_groups(S,A,l)
% clustering the node set using network state tensor S and adjacency matrix
% A.
%   S is a 3-dimensional sptensor: process x time x node
%   A is a symmetric sparse matrix: node x node
%   l is the number of clusters
%
%   clusV = find_node_groups(S,A) clusters the network-constrained graph G*
%   constructed from S & A using the PartGraphKway function from the Metis
%   library.
%   clusV is the returned vector containing group membership of nodes.

addpath('lib/tensor_toolbox/');
addpath('lib/tensor_toolbox/met/');
addpath('lib/metis-5.0.2/metismex-master/');

%% create network-constrained tensor graph G*
m = double(size(S,1));
q = double(size(S,2));
n = double(size(S,3));
numVertices = m * q + n;
[subs,vals] = find(S);
subs(:,1) = (subs(:,1)-1)*q + subs(:,2) + n;
[rows_A,cols_A,vals_A] = find(A);
gstar = sparse([subs(:,3);subs(:,1);rows_A;(1:n)'],...
    [subs(:,1);subs(:,3);cols_A;(1:n)'],...
    [vals;vals;vals_A;ones(n,1)],...
    numVertices,numVertices);

%% cluster the graph G*
options = struct();
options.wgtflag = true;
options.adjwgt = true;
clusall = metismex('PartGraphKway',gstar,l,options)+1;    

%% extract the cluster for nodes in V
clusV = clusall(1:n);