function [mu sigma2] = params_Gauss(X)
%   For use product/multivariate version of guassian distribution.
%   [mu sigma2] = params_pGuass(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector

% Useful variables
[m, n] = size(X);

mu = zeros(n, 1);
sigma2 = zeros(n, 1);

mu = mean(X);
%Option for N not N - 1
sigma2 = var(X,1);

end
