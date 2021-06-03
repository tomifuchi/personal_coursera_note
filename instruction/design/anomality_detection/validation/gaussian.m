function p = gaussian(X, mu, sigma2)
	p  = (1/(sqrt(2*pi*sigma2))) *...
	      exp((-(X - mu).^2)/(2*sigma2));
end
