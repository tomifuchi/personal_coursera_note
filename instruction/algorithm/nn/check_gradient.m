function [diff, numerical_grad] = check_gradient(nn_params, params_dim , X, y, lambda)
%Checking gradients to numerical
% Before running this, this is extremely slow way of calculating derivative.
% Besure to check which one you wanted to calclute, or check if recent_gradient_check.mat
% Exists, use that to calculate pre-existing.
% 
% Recent calcution is with the week_5 coursera course data, lambda 3. For crosschecking
% with the exercise it self.
% Accuracy rate: 1.66287e-08 #1 This
% Accuracy rate: 1.04734e-09 #2 Pentium 4

[J , grad] = nnCostf(nn_params, params_dim , X, y, lambda);

% Short hand for cost function
costFunc = @(p) nnCostf(p, params_dim , X, y, lambda);

epsilon = 1e-4;
theta = nn_params;
status = [];

for i = 1:length(theta)
        thetaPlus = theta;
        thetaPlus(i) += epsilon;
        thetaMinus = theta;
        thetaMinus(i) -= epsilon;
        gradApprox(i) = (costFunc(thetaPlus) - costFunc(thetaMinus))/(2*epsilon);
	status = [i gradApprox(i)]
endfor

diff =gradApprox';
%%Running maybe a hassle, save it after once
save recent_gradient_check.mat gradApprox;

%%This diff here to runs like fucking molases
numerical_grad = gradApprox';
diff = norm(numerical_grad-grad)/norm(numerical_grad+grad);
fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);
end
