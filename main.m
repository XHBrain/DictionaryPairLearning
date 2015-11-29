close all; clear; clc;

load handwrite;

%see the dataset
% for i = 1 : 10
%     displayData(X(:,1:20:500,i)');
% end

% lambda=0.3;
% tau=0.3;
% m=100;
% gamma = 10^-4;

m = 30;
tau    = 0.05;
lambda = 0.003;
gamma  = 0.0001;

[P D] = DictionaryPairLearning(X, lambda, tau, m, gamma);