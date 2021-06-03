function [theta, cost] = gd_fminunc(X, y, theta)
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(fcost(X, y, t)), theta, options);
end
