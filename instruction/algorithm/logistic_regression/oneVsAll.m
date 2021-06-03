function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Set Initial theta
initial_theta = rand(n + 1, 1);

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 1000);

% Run fmincg to obtain the optimal theta
% This function will return theta and the cost 

%num_labels is assumed to be a vector of unique(y)
for j = 1:length(num_labels)
  [all_theta(j,:)] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == num_labels(j)), lambda)), initial_theta, options);
endfor
% =========================================================================

end
