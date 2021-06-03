function p_x = productGaussian(X, mu, sigma2)
%Assuming you have obtained all the mu and sigma2 from 
%the param_Gauss.
[m n] = size(X);
p_x = [];

for i = 1:m
	p_x(i) = 1;
	for j = 1:n
		p_x(i) = p_x(i)*gaussian(X(i,j),mu(j),sigma2(j));
	endfor
endfor

p_x = p_x';

endfunction
