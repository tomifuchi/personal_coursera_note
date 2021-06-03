function moved_centroids = moveCentroids(X, c, K)
% INPUT
% X : data
% c : c(i) is the group of observation(i) or X(i,:) is the closest to 
% K : number of clusters count from 1.
% OUTPUT
% moved_centroid : moved from the mean of data assigned to k.

% Useful variables
[m n] = size(X);

% moved_centroids
moved_centroids = zeros(K, n);

% move the damn thing.
for k = 1:K
	moved_centroids(k,:) = mean(X(c==k,:));
end

end

