%% make leave-one-out cross validation prediction for process with ID pid.
% change pid to predict for another process.
% pid should be in the range [1,m], where m is the  total number of
% processes in the dataset.
% prediction output is plotted in file code/pred_<pid>.png

addpath('lib/tensor_toolbox/');
addpath('lib/tensor_toolbox/met/');
addpath('lib/metis-5.0.2/metismex-master/');
addpath('lib/poblano_toolbox/');

%% load data
load '../data/behance.mat';
%load '../data/twitter.mat';

% pid is the process to be predicted
pid = 6; 

%% prediction task
l = 10; % number of groups
t1 = 30; % training period is [0, t1]
t2 = q; % make prediction for time [t1+1, t2]
lda = 0.1; % regularization factor lambda
R = 50; % number of latent dimensions
k = 10; % number of topk

% whether to print the iterations of gradient descent
% 'final' : print last iteration
% 'off': do not print
% 'iter' (default): print everything
verbose = 'iter';

normdata = true; % whether to normalize data in factorization, 
                    % which may make learning faster and more accurate

%% make prediction for pid
% Note that this demo is slow since the node groups C and the group-based
% network tensors X (and the normalized version of X) need to be computed for  
% each chosen process pid in the leave-one-out cross-validation.
% Whereas, in our experiments in the paper, we only find C & construct X once for
% each fold in 5-fold cross-validation.
[pred,gtruth,error,C,X] = gpop(pid,S,A,l,k,t1,t2,R,lda,...
    'verbose',verbose,'normdata',normdata);
fprintf('Process %d. Error: %f\n', pid, error);


%% plot the prediction & ground-truth
% get data of training period
traindata = double(X(pid,1:t1,:));

% append prediction to training data for plotting
gtruth = [traindata;gtruth];
pred = [traindata;pred];

% get the limitation so that the plots have the same scale
maxy = max(sum(gtruth(end,:)),sum(pred(end,:)));
maxytrain = max(sum(traindata(end,:)));

% plot ground truth
clf()
subplot(1,2,1);
area(gtruth);
hold on;
plot([t1,t1],[0,maxytrain],'r','LineWidth',2);
ylim([0,maxy]);
ax = gca();
set(ax,'XTick',sort([ax.XTick,t1]))
title(strcat('Ground-truth. t1=',num2str(t1)));
xlabel('Time');
ylabel('Popularity');

% plot prediction
subplot(1,2,2);
area(pred);
hold on;
plot([t1,t1],[0,maxytrain],'r','LineWidth',2);
ylim([0,maxy]);
ax = gca();
set(ax,'XTick',sort([ax.XTick,t1]))
title(strcat('Prediction. t1=',num2str(t1)));
xlabel('Time');
ylabel('Popularity');

% save to file pred_pid.png
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 7 3];
print(strcat('pred_',num2str(pid)),'-dpng','-r0')