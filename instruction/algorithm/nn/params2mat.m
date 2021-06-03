function [theta] = params2mat(nn_params, params_dim)
	%Reshape into matrixes, stored in a cell array
	theta = {};
	j = 0;

	for i = 1:size(params_dim,1)
		theta(i) = {reshape(nn_params(j+1:j+prod(params_dim(i,:))),params_dim(i,:))};
		j = prod(params_dim(i,:));
	endfor
end
