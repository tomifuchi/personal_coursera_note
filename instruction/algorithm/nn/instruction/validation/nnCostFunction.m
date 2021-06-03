function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
%   This version truely works with any amount of theta's or input layer.

%Layer 2 Theta1
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

%Layer 3 Theta2
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%Given L, layer size, number of thetas are L-1
th = {Theta1;Theta2}; %PUt them into cell-array

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%===Feed forward=======
%Calculating a's Following this formula a = sigmoid(theta'*X)

%Activation units to prediction
[a,z] = forward_propagation(X,th);
h_theta = a{3};

%Y relabelling to 1's 0's
y_relabeled = relabelling(y);

%===Cost function=======
%In summary, calculate then sum all the cost of the output units. Nothing special
%Not including biased

unrolled_thetas = unroll_theta(th);
J =  (-1/m) * ...
  sum(sum((times(y_relabeled,log(h_theta)) + times((1 - y_relabeled),log(1 - h_theta))), 1)) + (lambda/(2*m))*sum(unrolled_thetas.^2);

%%====Back propagation====
%Calculating the error in all layer except input layer, denoting delta, then 
%use delta to calculate gradients
g = back_propagation(a,z,th,y,lambda);
Theta1_grad = g{1};
Theta2_grad = g{2};

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
