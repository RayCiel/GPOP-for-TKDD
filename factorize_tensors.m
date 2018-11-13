function [Fac, objfunc] = factorize_tensors(traindata,R,lda,varargin)
%   traindata contains traindata.T,traindata.Y, traindata.Z, traindata.Q
%   R: number of latent dimensions
%   lda: regularization factor
% lda : lambda
%
%   'verbose' - whether to print each iteration of the gradient descent
%   algorithm, can be either one three following strings:
%       'final' : only print the final iteration
%       'off' : do not print
%       'iter' : print each iter
%
%   [Fac, objfunc] = factorize_tensors(Z,R,lda,varargin)
%       Fac: decomposition, a struct contains D,H,J,F,K
%       objfunc : objective function value
%


%% Set parameters
params = inputParser;
params.addParameter('verbose', 'iter', @(x) ismember(x,{'iter','final','off'}));
params.addParameter('normdata',false,@islogical); % whether to normalize data 
params.parse(varargin{:});

%% Set up optimization algorithm options
options = ncg('defaults');
if ~isempty(params.Results.verbose)
    options.Display = params.Results.verbose;
end     

%% prepare data for coupled tensor decomposition
% coupled tensor decomposition, using both group size & total size
% Topk results for groups (A,B,C) & total (D,E,F) are computed separately, 
% Six tensors: Z.object = {A,B,C,D,E,F}
% Dimensions: G,P1,P2,P3,T1,T2,Gsum
% Dimensions: Z.size = [numClus,numTrainP1,numTrainP2,numTestP,numTrainT,numTestT,1];
% Modes: Z.mode = {[1,2,5],[1,2,6],[1,4,5],[7,3,5],[7,3,6],[7,4,5]};
% Prediction would be: [1,4,6]
% Dimension 7 = sum of row of dimension 1

Z = traindata.Z; Y = traindata.Y; T = traindata.T; Q = traindata.Q;
numClus = size(T,3);
numTrainPgroup = size(T,1);
numTrainPsum = size(Y,1);
t1 = size(Z,2);
t2 = size(T,2);
trainT = 1:t1;
testT = t1+1:t2; 
numTrainT = t1;
numTestT = t2-t1;

%% convert data to change from t1
for t = t1+1:t2
    T(:,t,:) = T(:,t,:)-T(:,t1,:);
    Y(:,t,:) = Y(:,t,:)-Y(:,t1,:);
end

%% prepare data for cmtf
if nnz(T) == 0
    A = sptensor([numClus,numTrainPgroup,numTrainT]);
    B = sptensor([numClus,numTrainPgroup,numTestT]);
else
    [subs,vals] = find(T);
    idx = subs(:,2) <= t1;
    if any(idx>0)
        A = sptensor(subs(idx,[3,1,2]),vals(idx),[numClus,numTrainPgroup,numTrainT]);
    else
        A = sptensor([numClus,numTrainPgroup,numTrainT]);
    end
    idx = subs(:,2) > t1;        
    if any(idx>0)
        subs(:,2) = subs(:,2) - t1;
        B = sptensor(subs(idx,[3,1,2]),vals(idx),[numClus,numTrainPgroup,numTestT]);
    else
        B = sptensor([numClus,numTrainPgroup,numTestT]);
    end
end

if nnz(Y) == 0
    D = sptensor([1,numTrainPgroup,numTrainT]);
    E = sptensor([1,numTrainPgroup,numTestT]);
else
    [subs,vals] = find(Y);
    idx = subs(:,2) <= t1;
    if any(idx>0)
        D = sptensor(subs(idx,[3,1,2]),vals(idx),[1,numTrainPgroup,numTrainT]);
    else
        D = sptensor([1,numTrainPgroup,numTrainT]);
    end
    idx = subs(:,2) > t1;
    if any(idx>0)
        subs(:,2) = subs(:,2) - t1;
        E = sptensor(subs(idx,[3,1,2]),vals(idx),[1,numTrainPgroup,numTestT]);
    else
        E = sptensor([1,numTrainPgroup,numTestT]);
    end
end

if nnz(Z) == 0
    C = sptensor([numClus,numTrainT,1]);
else
    [subs,vals] = find(Z);
    C = sptensor(subs(:,[3,1,2]),vals,[numClus,1,numTrainT]);
end

if nnz(Q) == 0
    F = sptensor([1,numTrainT,1]);
else
    [subs,vals] = find(Q);
    F = sptensor(subs(:,[3,1,2]),vals,[1,1,numTrainT]);
end

Z = struct();
Z.modes = {[1,2,5],[1,2,6],[1,4,5],[7,3,5],[7,3,6],[7,4,5]};
Z.size = [numClus,numTrainPgroup,numTrainPsum,1,numTrainT,numTestT,1];
Z.object{1} = tensor(A);
Z.object{2} = tensor(B);
Z.object{3} = tensor(C);
Z.object{4} = tensor(D);
Z.object{5} = tensor(E);
Z.object{6} = tensor(F);  

%% normalize the data
maxval = 1;
if params.Results.normdata 
    maxval = 0;
    for i = 1:length(Z.object)
        maxval = norm(Z.object{i})^2;
    end
    maxval = sqrt(maxval);

    for i = 1:length(Z.object)
        Z.object{i} = Z.object{i}/maxval;
    end
end

%% Initialization
sz = Z.size;
N = length(sz)-1;
G = cell(N,1);
for n=1:N
    G{n} = randn(sz(n),R);
    for j=1:R
        G{n}(:,j) = G{n}(:,j) / norm(G{n}(:,j));
    end
end  

%% Fit CMTF using Optimization
P = numel(Z.object);
Znormsqr = cell(P,1);
for p = 1:P
    Znormsqr{p} = norm(Z.object{p})^2;
end

%% decompose
out = feval(@ncg, @(x)compute_fg(x,Z,lda,...
    Znormsqr), tt_fac_to_vec(G), options);

%% Compute factors and model fit
P = ktensor(vec_to_fac(out.X, Z));
objfunc = out.F;

%% Change the format to match the writing in the paper
Fac = struct();
Fac.F = P.U{1};
Fac.D = P.U{2};
Fac.H = P.U{3};
Fac.K = P.U{4};
Fac.J = [P.U{5};P.U{6}];
% change prediction to accumulative values, instead of change from t1
for t = t1+1:t2
    Fac.J(t,:) = Fac.J(t,:) + Fac.J(t1,:);
end

%% recover true value if normalized earlier
if params.Results.normdata
    Fac.J = Fac.J * maxval; 
end

function [f,g] = compute_fg(x, Z, lda, Znormsqr)
% computes the function value and the gradient for coupled tensor
% decomposition and change

%% Convert the input vector into a cell array of factor matrices
A  = vec_to_fac(x,Z);

% a factor for sum of group
A{end+1} = sum(A{1});

P = numel(Z.object);

fp = cell(P,1);
Gp = cell(P,1);
numClus = Z.size(1);
for p = 1:P
    [fp{p},Gp{p}] = tt_cp_fg(Z.object{p}, A(Z.modes{p}), Znormsqr{p});    
end

% restore the number of factors
A = A(1:end-1);

%% Compute overall gradient
G = cell(size(A));
for n = 1:numel(G)
    % regularization gradient
    G{n} = lda * A{n};
end
for p = 1:P
    for i = 1:length(Z.modes{p})
        j = Z.modes{p}(i);
        if j <= length(G)
            G{j} = G{j} + Gp{p}{i};
        else % sum of group
            G{1} = G{1} + repmat(Gp{p}{i},numClus,1);
        end
    end
end

%% Compute regularization part of objective function
reg_f = 0;
for n = 1:numel(G)
    reg_f = reg_f + norm(A{n},'fro')^2;
end

%% Compute overall function value
f = sum(cell2mat(fp)) + 0.5 * lda * reg_f;

%% Vectorize the cell array of matrices
g = tt_fac_to_vec(G);

function A = vec_to_fac(x,Z)
% CMTF_VEC_TO_FAC Converts a vector to a cell array of factor matrices
% The last factor in Z is ignored (it is the sum of factor 1

%% Set-up
P   = length(x);
sz  = Z.size;
N   = length(sz)-1;

%% Determine R
R = P / sum(sz(1:N));

%% Create A
A = cell(N,1);
for n = 1:N
    idx1 = sum(sz(1:n-1))*R + 1;
    idx2 = sum(sz(1:n))*R;    
    A{n} = reshape(x(idx1:idx2),sz(n),R);
end

