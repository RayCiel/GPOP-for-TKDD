addpath('lib/tensor_toolbox/');
addpath('lib/tensor_toolbox/met/');
load '../data/behance.mat';

%% running time vs. l (#groups)
time_clus_vs_l = [];
all_ls = 5:5:50;
for i = 1:length(all_ls)
    l = all_ls(i);
    tic;
    clusV=find_node_groups(S,A,l);
    time_clus_vs_l(i) = toc;
    time_clus_vs_l
end

%% running time vs. m (#processes)
time_clus_vs_m = [];
l = 12;
for i = 1:13
    m = i*100;
    tmpS = S(1:m,:,:);
    tic;
    clusV=find_node_groups(tmpS,A,l);
    time_clus_vs_m(i) = toc;
    time_clus_vs_m
end

%% running time vs. n (#num nodes)
time_clus_vs_n = [];
l = 12
for i = 1:8
    n = 10000*i;
    tmpA = A(1:n,1:n);
    tmpS = S(:,:,1:n);
    tic;
    clusV=find_node_groups(tmpS,tmpA,l);
    time_clus_vs_n(i) = toc;
    time_clus_vs_n
end
