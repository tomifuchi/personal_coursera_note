function [err] = pca_err(S,k)
%Calculating variance ratio for the pca dimension reduction algorithm.
%We will be expecting a diagonal matrix S from svd to be passed here
	err = sum(diag(S)(1:k))/sum(diag(S)(:));
end
