function [X_reduced, U_reduced, S] = pca_mine(X, k)
%PCA Run principal component analysis on the dataset X
%   X assumed has precossed data
%   [X_reduced, U, S] = pca(X) reduced down X from R^n to R^k with k preferably
%   significantly lesser than n

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% Covariance matrix
Sigma = (1/m) * X' * X; %nxn matrix

% Eigen vector, singular value decomposition.
[U, S, V] = svd(Sigma); %mxn matrix

%We take the U which is nxn, to reduce to k, take first k collumn of U                  
U_reduced = U(:,1:k); %nxk

%To calculate X reduced down from R^n to R^k
X_reduced = X * U_reduced;

end
