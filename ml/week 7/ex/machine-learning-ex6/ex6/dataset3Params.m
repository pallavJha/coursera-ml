function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels ons the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_all = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_all = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
meanPredictions = zeros(size(C_all,1), size(sigma_all,1));

for i = [1:size(C_all,1)]
    for j = [1:size(sigma_all,1)]
        model= svmTrain(X, y, C_all(i), @(x1, x2) gaussianKernel(x1, x2, sigma_all(j)));
        predictions = svmPredict(model, Xval);
        meanPredictions(i, j) = mean(double(predictions ~= yval));
    endfor
endfor

#disp(meanPredictions);
[C_min_index sigma_min_index] = find(min(min(meanPredictions)) == meanPredictions);
C = C_all(C_min_index);
sigma = sigma_all(sigma_min_index);
% =========================================================================

end
