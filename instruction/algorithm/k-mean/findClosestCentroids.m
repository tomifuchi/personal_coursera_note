function c = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   c = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   
%   INPUT
%   X: data
%   centroids: positions of centroid.
%   OUTPUT
%   c: vector of mx1, each element correspond to i observation of X or X(i,:) closest 
%   a centroid k.
%   
%   Note:
%   This step is step 1 in the k-mean algorithm.
%   This step is also used to predict the data, with given optimized centroid and
%   unknown data.

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
c = zeros(size(X,1), 1);

%A centroid is an example aswell, just so you remember so it should be
m = size(X,1);

%A distance matrix of centroids
%K collumn for centroid, m row of observations
%m to k is, distance from m observation to k centroid
dist = zeros(m,K);

%For k centroids
for i = 1:m
    for k=1:K 
       dist(i,k) = (norm(X(i,:) - centroids(k,:)))^2;
endfor
[small_dist, c] = min(dist, [], 2);

end
