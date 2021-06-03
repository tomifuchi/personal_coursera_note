function [F1, P, R, true_p, true_n, false_p, false_n] = calF1(y_pred, y, display_score=false)

%Calculating F1 score for evaluation, this will work for just
%about any problem or any algorithm.
% y and y_pred is mx1

%Appending into a matrix for indexing
D = [y_pred y];

%Finding true/false positives/negatives
%Indexing every row of first collumns thats is one, then index that matrix
%collumn 1 == collumn 2
%true postives, true negatives
true_p = sum(D(D(:,1)==1,:)(:,1) == D(D(:,1)==1,:)(:,2));
true_n = sum(D(D(:,1)==0,:)(:,1) == D(D(:,1)==0,:)(:,2));
%false postives, false negatives
false_p = sum(D(D(:,1)==1,:)(:,1) != D(D(:,1)==1,:)(:,2));
false_n = sum(D(D(:,1)==0,:)(:,1) != D(D(:,1)==0,:)(:,2));

%Precision: (True positive)/(True postive + False positive). (Accuracy, high chance
%of it's something)
%Recall: (True positive)/(True pos + False negatively = Actual positive in the
%data set). (Detection in real data)

%この関数は馬鹿者を壊れないように
if true_p == 0 && (false_p == 0 || false_n == 0)
	F1 = 0;
	return 
endif
	
%Precision
P = true_p/(true_p + false_p);

%Recall
R = true_p/(true_p + false_n);

%Formula for F1 score
F1 = (2*P*R)/(P+R);

if display_score == true
	fprintf("True_p : %i, False_p: %i\nFalse_n: %i , true_n: %i\nPrecision: %f, Recall: %f\nF1 score: %f\n",true_p, false_p, false_n, true_n, P, R, F1);
end

endfunction
