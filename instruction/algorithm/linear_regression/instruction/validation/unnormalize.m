function [unnormalized_X] = unnormalized(X_norm)
%Assumed unbiased X
mu = mean(X_norm); %returns a 1x2
sigma = std(X_norm); % ||

unnormalized_X  = (X_norm*sigma) + mu;

end
