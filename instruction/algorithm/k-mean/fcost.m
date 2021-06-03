function J = fcost(X, centroids, c)

% Cost function for k-mean algorithm

%initialized J
J = 0;

% Number of observations 
m = size(X,1);

% Calculting the cost
for i = 1:m
	J += norm(X(i,:) - centroids(c(i),:))^2;
end

% Divided by 1/m
J = 1/m*J;

end
