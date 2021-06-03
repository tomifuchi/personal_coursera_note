function [theta, cost] = gd_fminunc(X, y, theta, numiters)
options = optimset('GradObj', 'on', 'MaxIter', numiters);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(fcost(X, y, t)), theta, options);
end
