function [J grad] = nnCostf(nn_params, params_dim , X, y, lambda)

%   Cost and grad for neural network
%   Input:
%   * nn_params: Unrolled version of all theta's, assumed randomly initialized
%   this need to be converted back into the weight matrices. 
%   * params_dim: A length(theta) x 2 matrix, with each row represent dimension
%   of each theta, ascending. col 1 for rows, col 2 for collumns
%   * num_labels: unique(y);
%   * X: Input matrix, not biased.
%   * y: output
%   % lamdbda: regularization part, 1, 2, 10, or 0.1 or 0.01. Experiment with this
%   so the data will not over fit
%
%   Steps:
%   * Reshape params to theta matrixes in cellarray 
%   * Forward propagation to obtain activation units and z=a*theta.
%   * Relabelling if y is not binary
%   * J
%   * Backpropagation obtaining g, gradients
%   * Cross-checking gradient using check_gradient (not yet implimented)
%
%   Output:
%   J : cost 
%   grad: gradient of costfunction unrolled. Rerolled by using the params2mat

% Reshapeing matrix
theta = params2mat(nn_params, params_dim);

% Setup some useful variables
m = size(X, 1);
L = size(params_dim,1) + 1; %number of layer
K = unique(y);
   
% Set J to 0
J = 0;

%===Feed forward=======
%Calculating a's Following this formula a = sigmoid(theta'*X)

%Activation units to prediction
[a,z] = forward_propagation(X,theta);
h_theta = a{L};

%Y relabelling to 1's 0's, no relabelling if binary
if(!isequal(K,[0;1]))
	y_relabeled = relabelling(y);
else
	y_relabeled = y;
endif

%===Cost function=======
%In summary, calculate then sum all the cost of the output units. Nothing special
%Not including biased

unrolled_thetas = unroll_theta(theta);

%Cost function w/wout regularization is already checked with another example
J =  (-1/m) * ...
  sum(sum((times(y_relabeled,log(h_theta)) + times((1 - y_relabeled),log(1 - h_theta))), 1)) + (lambda/(2*m))*sum(unrolled_thetas.^2);

%%====Back propagation====
%Calculating the error in all layer except input layer, denoting delta, then 
%use delta to calculate gradients
g = back_propagation(a,z,theta,y,lambda);

% Unroll gradients
grad = unroll_grads(g);

end
