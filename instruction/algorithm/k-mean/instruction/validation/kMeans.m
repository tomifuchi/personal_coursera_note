function [centroids, c, cost] = kMeans(X, initial_centroids, ...
                                      max_iters)
% K-mean algorithm to classify, unlabled data, to find some structure to it.
% Find k clusters.

% INPUT:
% X: data
% initial_centroids: initialized centroid preferably randomized, X data positions.
% max_iters: maximum number of iterations
% OUTPUT
% centroids: the centroid in the data optimized.
% c: mx1 with each element represent which group does X(i,:) is the closest to. Should
% be from 1 to K

% Initialize values
[m n] = size(X);
K = size(initial_centroids, 1);
centroids = initial_centroids;
previous_centroids = centroids;
c = zeros(m, 1);

cost =[];

% Run K-Means
for i=1:max_iters

    fprintf('K-Means iteration %d/%d...\n', i, max_iters);
    
    % For each example in X, assign it to the closest centroid
    c = findClosestCentroids(X, centroids);
    
    % Given the memberships, compute new centroids
    centroids = moveCentroids(X, c, K);

    % Cost
    cost(i) = fcost(X, centroids, c);
    fprintf('Cost: %.10f\n',cost(i));

    % Larger will not happens, it's impossible, mathematically saying
    % If it's equal then it's converge, should stop to save time.
    if i > 1 && cost(i) >= cost(i-1)
	    break;
end

end
