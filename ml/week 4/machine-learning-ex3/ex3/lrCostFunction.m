function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
hx = X * theta;

#disp("Displaying hx1");
#disp(hx);

hx = sigmoid(hx);	

#disp("Displaying after sigmoid hx2");
#disp(hx);

hx1 = log(hx);

hx2 = log(1 - hx);

hx1 = y' * hx1;

oppositeY = abs(1 - y);

hx2 = oppositeY' * hx2;

J = hx1 + hx2;

J = -J / m;

hxydiff = hx - y;

sqTheta = theta.^2;
#regSum = lambda * (sum(sqTheta) / (2 * m));
#disp("sqTheta")
#disp(sqTheta)

#disp("regsum")
#disp(regSum)
sqTheta(1) =  0;
regSum = lambda * (sum(sqTheta) / (2 * m));
#disp("sqTheta")
#disp(sqTheta)

#disp("regsum")
#disp(regSum)
J = J + regSum;

#disp("Displaying hx");
#disp(hx);

#disp("Displaying y");
#disp(y);

[r loopLength] = size(X);
grad = zeros(loopLength, 1);

for i = 1:loopLength
	leftX = X(:,i);
	grad(i) = (hxydiff' * leftX) / m;
	if (i != 1)
		grad(i) = str2double(sprintf("%.4f", grad(i) + (lambda * theta(i) / m)));
	endif
endfor

% =============================================================

end
#grad = grad(:);

#end
