function [X_reduced] = pca_reduce(X, U_reduced)
%PCA Run principal component analysis on the dataset X
%   X data
%   [X_reduced] = pca_reduce(X) reduced down X from R^n to R^k with k preferably
%   significantly lesser than n

%To calculate X reduced down from R^n to R^k
X_reduced = X * U_reduced;

end
