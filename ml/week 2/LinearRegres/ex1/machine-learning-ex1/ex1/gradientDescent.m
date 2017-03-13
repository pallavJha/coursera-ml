function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    #J_history(iter) = computeCost2(X, y, theta);
    #fprintf("J_history(iter) = %f", J_history(iter));
    theta1 = theta(1) - (alpha * computeCost2(X, y, theta, 1));
    theta2 = theta(2) - (alpha * computeCost2(X, y, theta, 2));
    theta(1) = theta1;
    theta(2) = theta2;
    %fprintf("J_history(iter) = %f, theta1 = %f, theta2 = %f\n", computeCost(X, y, theta), theta1, theta2);
    % ============================================================

    % Save the cost J in every iteration    


end

end


function J = computeCost2(X, y, theta, xNo)
m = length(y); % number of training examples

J = 0;

H = X * theta;

Z = H - y;

onlyX = X(:,xNo)';

#Z = Z.^2;

J = onlyX * Z; 

J = J / m;

% =========================================================================

end
