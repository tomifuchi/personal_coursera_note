function [theta, J_history] = gd(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
%   Some hints 
%   alpha: 0.01, 0.1, 0.001,...
%   num_inter: 400, 500, 1000 choice on you

% Initialize some useful values
X = [ones(size(X,1),1) X]; %Assumed unbiased
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
   predictions = X*theta; %Getting the hypothesis y's
   theta = theta - alpha*(1/m) * ((predictions - y)'*X)';
   % Save the cost J in every iteration    
   J_history(iter) = fcost(X(:,2:end), y, theta);
end

end
