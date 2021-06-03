function [a,z] = forward_propagation(X,theta)
%For internal use with the nncostfunction
%Forward propagation, for neuron network
% 
% Assumed:
% X has no biased unit added
% theta are a cell array
%
% Return value
% a: return cell array of activation unit, a(i) is activation unit at layer i
% Prediction to be classify to group is at a(l) or the last activation unit.
% z: return the (biased a *theta), with z(i) on computer is z at layer i+1 on paper
% z should start at 2 but on computer z start at 1,i.e z(1) is z_2 on paper
%

l = length(theta); %Number of a's needed to be calculated

% Forward propagation
% Return activation units as cell array of vectors collumn wise
a = {}; z = {};

%At layer 1
a(1) = {X}; %Making a_1

%Calculating layer 2 to end
for i = 1:l
	z(i) = {[ones(size(a{i},1),1) a{i}]*theta{i}'};
	a(i+1) = {sigmoid(z{i})};
endfor 
end
