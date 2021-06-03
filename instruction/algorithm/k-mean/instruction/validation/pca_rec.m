function [X_rec] = pca_rec(X_reduced, U_reduced)
% Assuming running PCA you will have U_reduced = U(:,1:k)
% Then to approximate back what X used to be 

X_rec = X_reduced * U_reduced'; %X_rec will be close to the original X                  
end
