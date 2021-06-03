function [grad] = back_propagation(a,z,theta,y,lambda)
	% Back propagation algorithm
	% To calcluate gradient of the cost function J
	% 
	% Assumed 
	% * a is a cell array of a's including a1, all with no biases.
	% * z is a cell array of a*theta's, z(i) on computer is z at i+1 layer on paper
	% * y not relabeled

	grad={};

	%Relabelling y
	K = unique(y);
	if(!isequal(K,[0;1]))
		y_relabeled = relabelling(y);
	else
		y_relabeled = y;
	endif

	%Initialized delta
	delta = {};
	l = length(a); %number of theta's = delta's
	m = size(a{1},1); %Not passing X in so...

	%Find the delta_L to delta_2, in this vector
	delta(1) = {0}; %No delta1
	delta(l) = {a{l} - y_relabeled}; %Last layer

	%This looks complicated, but try to look at it carefully,
	%Look at i
	%Calculating deltas
	for i = l-1:-1:2
		delta(i) = {delta{i+1}*theta{i}(:,2:end) .* sigmoidGradient(z{i-1})};
	endfor

	%Calculating gradients with regularization
	%Add biases to a's
	for i = 1 : l-1
		grad(i) = {(1/m)  * (([ones(size(a{i},1),1) a{i}]' * delta{i+1})'...
			+ lambda*[zeros(size(theta{i},1),1) theta{i}(:,2:end)])};
	endfor
end
