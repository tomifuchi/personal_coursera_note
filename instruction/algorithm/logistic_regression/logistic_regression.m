function p = logistic_regression(X,y,num_label,lambda,pred)
%Plug everything in and go to town
%Output is 1 2 coresspond to the index of unique(y)
p = predictOneVsAll(oneVsAll(X,y,num_label,lambda),pred, num_label);
