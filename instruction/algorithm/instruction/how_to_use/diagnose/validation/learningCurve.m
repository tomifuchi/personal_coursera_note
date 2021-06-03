function [error_train, error_cv] = ...
	learningCurve(X_train, y_train, X_cv, y_cv, lambda, interval=1)
% Learning curve Generates the train and cross validation set errors needed 
%
%   Input:
%   fcost(X,y,theta,lambda)
%   X_train, y_train of train set
%   X_cv, y_cv of cross-validation set
%   lambda: Regularization term
%   interval: step when going through m (Default is 1)
%   
%   m in practice can sometimes set into larger interval
%   Output:
%  
%   error_train: cost of train set as train set size increase
%   error_cv: cost of cross-validation set as train set size increase

% Number of training examples
m = size(X_train, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% X and y for training set
% X_cv and y_cvfor cross-validation set
% 
% Different algorithm change train and cost fnnction,keep argument.
% 
% The format for train should be
% train...(X,y,initial_theta,lambda)
% and cost function
% 
for i = 1:m
	op_theta = trainLinearReg(X_train(1:i,:),y_train(1:i),lambda);
	error_train(i) = linearRegCostFunction(X_train(1:i,:),y_train(1:i),op_theta, 0);
	error_cv(i)= linearRegCostFunction(X_cv,y_cv,op_theta, 0);
endfor

%%Plot if needed
%plot(1:m, error_train, 1:m, error_cv);
%title('Learning curve')
%legend('Train', 'Cross Validation')
%xlabel('Number of training examples')
%ylabel('Error')
%
%%Print if needed
%fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
%for i = 1:m
%    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
%end

end
