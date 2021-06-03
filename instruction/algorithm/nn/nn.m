%Edit this file to run multiple times if needed
%For lean, mean application of the neuron network

%X and Y
X = [0 0;0 1;1 0;1 1];
y = [1;1;0;1];
%Theta's
theta1= randInitializeWeights(2,2);
theta2= randInitializeWeights(2,1);
theta = [theta1(:);theta2(:)];

%Theta's dim
params_dim = [size(theta1);size(theta2)];

%Lambda
lambda = 0; %Logical doesn't overfit

%Iteration num
max_iter= 70;

options = optimset('GradObj', 'on', 'MaxIter', max_iter);                                     
%If you need 1 time
%[grad cost] = fmincg(@(p) nnCostf(p, params_dim, X, y, lambda), theta, options);        

%For loop to find the smallest
min_cost = 1;
min_grad = [];
j = 0;
for i = 1:10
	%Randomly initialized again
	theta1= randInitializeWeights(2,2);
	theta2= randInitializeWeights(2,1);
	theta = [theta1(:);theta2(:)];

	disp(i);
	[grad cost] = fmincg(@(p) nnCostf(p, params_dim, X, y, lambda), theta, options);
	if(cost(end) < min_cost(end))
		j = i;
		min_grad = grad;
		min_cost = cost;
	endif
endfor
fprintf("Lowest cost iteration is %i\n",j);

%Predict %1
r = params2mat(min_grad,params_dim);
forward_propagation(X,r)

%%Predict %1
%r = params2mat(grad,params_dim);
%forward_propagation(X,r)
