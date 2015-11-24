function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


h = sigmoid(X * theta);
h_y = -y .* log(h);

h_c = 1 - h;
h_y_c = -((1 - y) .* log(h_c));

J = sum(h_y + h_y_c) / m;

[row, col] = size(theta);
theta_sq = (sum(theta(2:row,col) .^ 2)  * lambda )/ (2 * m);
J = J + theta_sq;

theta_cl = theta * (lambda / m);
theta_cl(1,1) = 0;

diff =  h - y;
grad_reg = ((diff' * X)/m)';

grad = grad_reg + theta_cl;
% =============================================================

end
