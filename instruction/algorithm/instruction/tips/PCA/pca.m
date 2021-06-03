function [U_reduced, S] = pca(X, k)
%PCA Run principal component analysis on the dataset X
%   X assumed has precossed data
%   [U_reduced , S] = pca(X) reduced down X to obtained eigen vector 
%   then you can from R^n to R^k with k preferably 
%   significantly lesser than n using pca_reduce or pca_recover.

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

end
