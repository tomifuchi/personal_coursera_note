function p = predict(theta,X)
	p = [ones(size(X,1),1) X]*theta;
end
