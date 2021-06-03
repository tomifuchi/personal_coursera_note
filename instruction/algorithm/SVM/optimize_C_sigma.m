function [C, sigma] = optimize_C_sigma(X_train, y_train, X_cv, y_cv)

% Takes in X_train y_train set, and cv set. 
% Train on a set of C and sigma, do it for all values in C_vec and sigma_vec
% selects C and sigma with smallest error_cv

% Changed if needed
C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

%Length
m = size(X,1);

%Error train and error val
%C collumn x  sigma row
error_train = zeros(length(sigma_vec),length(C_vec)); 
error_cv = zeros(length(sigma_vec),length(C_vec));

%Find C and sigma
%Change the svmtrain and svmPredict if needed for a custom SVM
for i = 1: length(C_vec)
	for j = 1:length(sigma_vec)
		fprintf("\nTesting C: %f, Sigma: %f\n",C_vec(i), sigma_vec(j))
		fprintf("========================================\n");
		model = svmTrain(X, y, C_vec(i), ...
		@(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));

		%Train set is not necessarcy but what the hell
                predictions_train = svmPredict(model, X);
                predictions_val = svmPredict(model, Xval);

		error_train(j,i) = mean(double(predictions_train ~= y));
		error_cv(j,i) = mean(double(predictions_val ~= yval));

		fprintf("error_train: %f\n", error_train(j,i));
		fprintf("error_cross: %f\n", error_cv(j,i));
	endfor
endfor

[smallest_row smallest_row_idx] = min(error_cv);
[smallest_col smallest_col_idx] = min(smallest_row);

C = C_vec(smallest_col_idx);
sigma = sigma_vec(smallest_row_idx(smallest_col_idx));

%Print if needed
%fprintf("best C:%f , best sig: %f\n", C, sigma);

end
