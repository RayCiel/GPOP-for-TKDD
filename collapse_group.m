function X = collapse_group(S,C,l)
% aggregated data in state tensor S into group-based tensor X
%   S is a 3-dimensional sptensor: process x time x node
%   C is a vector containing the group membership of nodes
%   l is the number of groups in C. if missing, l = max(C);
if ~exist('l','var')
    l = max(C);
end
[subs,vals] = find(S);
subs(:,3) = C(subs(:,3));
X = sptensor(double(subs),double(vals),double([size(S,1),size(S,2),l]));