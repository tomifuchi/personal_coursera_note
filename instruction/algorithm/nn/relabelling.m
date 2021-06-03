function y = relabelling(y)
  %This function simply relabels the y to this format unique(y), for classification
  %Problem .this will becomes a matrix or a vector depending on the quantity
  %of data passed in.
  %
  %Input: y is a vector of prediction in groups as an integers 1-n
  %Output: y is a matrix with collumns is the number of class group
  %Row as observations. 0 denote not in that group, 1 denote in that group

  %Example 
  %1 2 3 4
  %-------
  %0 0 1 0
  %1 0 0 0

  %Observation #1 is in group 3
  %Observation #2 is in group 1
  
  K=unique(y);
  temp = zeros(size(y), size(K));
  for i=1:length(y)
	 temp(i,y(i)) = 1;
  endfor
  y = temp;
end
