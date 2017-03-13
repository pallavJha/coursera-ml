function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
#disp("X");
#disp(X);
h = X * theta;
#disp("h");
#disp(h);
J = sum((h - y).^2);
J = J / (2 * m);

sqTheta = theta;
sqTheta = sqTheta.^2;
sqTheta(1) = 0;
sqThetaSum = sum(sqTheta);
regSum = sqThetaSum * lambda / (2 * m);

J = J + regSum;

for i = 1:size(theta)
#disp("X(:,i)")
#disp(X(:,i));
#disp("h'")
#disp(h');
	grad(i) = sum((h - y)' * X(:,i));
#disp("grad(i)")
#disp(grad(i));
	grad(i) = grad(i) / m;
#disp("grad(i) / m")
#disp(grad(i));
	if(i != 1)
		grad(i) = grad(i) + (lambda * theta(i) / m);
#		disp("grad(i) in regularization")
#		disp(grad(i));
	endif
endfor

% =========================================================================

grad = grad(:);

end
