function [J,grad] = fcost(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   Assumed X not biased
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
X = [ones(size(X,1),1) X];

predictions = X*theta;
J = (1/(2*m))*sum((predictions - y).^2);
grad = (1/m) * ((predictions - y)'*X);
end
