%----------------------------------------------------------------------
%                       One-layer Cost Function

function [J grad] = nnCostFunction1Hidden(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% --------------------------------  part 1 -------------------------------%
% pad one to the input X
tX = [ones(m,1) , X]; 

% input layer to hidden layer
activ = sigmoid(tX * Theta1');

% pad one the htheta_1
activ = [ones(m,1), activ];

% hidden layer to output layer
out = sigmoid(activ * Theta2');

% reformat y
rep_y = zeros(m, num_labels);
for i = 1:m
   rep_y(i, y(i)) = 1; 
end

J = sum(sum(-rep_y.* log(out)-(1-rep_y).*log(1-out))) / m;

% ensure the first column of theata are zeros
Theta1(:,1) = 0;
Theta2(:,1) = 0;

reg_term = lambda / (2 * m) * ( sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)) );

J = J + reg_term;


% --------------------------------  part 2 -------------------------------%

% Notes:
% y is remapped to rep_y
% activation on output layer is out
% activation on hidden layer is activ

% delta measures how much that a node was "responsible" for the erros in
% the output

delta_3 = out - rep_y;

gz2 = activ .* (1 - activ);
delta_2 = (delta_3 * Theta2) .* gz2;

% compute for theta1_grad
% accumulate the gradient
delta_2 = delta_2(:,2:end); % get rid of the 1st column

for i = 1:m
    cur_alpha = tX(i,:);
    cur_delta = delta_2(i,:);
    Theta1_grad  = Theta1_grad  + cur_delta' * cur_alpha;
end

Theta1_grad = Theta1_grad ./ m;

% compute for theta2_grad
for i = 1:m
    cur_alpha = activ(i,:);
    cur_delta = delta_3(i,:);
    Theta2_grad  = Theta2_grad  + cur_delta' * cur_alpha;
end

Theta2_grad = Theta2_grad ./ m;


% --------------------------------  part 3 -------------------------------%

% regulization for gradients
reg_theta1 = (lambda / m ) .* Theta1;
reg_theta2 = (lambda / m ) .* Theta2;

Theta1_grad = Theta1_grad + reg_theta1;
Theta2_grad = Theta2_grad + reg_theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
