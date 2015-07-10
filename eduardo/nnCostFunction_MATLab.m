%----------------------------------------------------------------------
%                       Test for Multilayer Cost Function
%
%   The multilayer version is set to 3 and 4 layers and compared to a three layer
%   version.

layer_num=4; % Number of layers

exNum=10; % Number of random examples
rangeInt=[5 100]; % Range for radom integer numbers vary

m=randi(rangeInt,exNum,1); % Random number of samples
input_layer_size=randi(rangeInt,exNum,1); % Random size for the input layer
hidden_layer_size=randi(rangeInt,exNum,layer_num-2); % Random size for the hidden layer
num_labels=randi(rangeInt,exNum,1); % Random number of labels


% X, and y are matrix cell arrays of exNum size
% each value is randomly generate
X=cell(exNum,1);
for k=1:exNum
    X{k}=rand(m(k),input_layer_size(k));
end

y=cell(exNum,1);
for k=1:exNum
    y{k}=randi([1 num_labels(k)],m(k),1);
end

lambda=rand(exNum,1); % Random lambdas

% Arrays for both costs and cell for both grads are created
J1=zeros(exNum,1);
J2=zeros(exNum,1);
grad1=cell(exNum,1);
grad2=cell(exNum,1);

for n=1:layer_num-2
% Both cost functions are run using the same random arguments.
% nnCostFunction is the general cost funtion and nnCostFunction1Hidden and 
% nnCostFunction2Hidden are cost functions for 1 and 2 hidden layers,
% respectively.

if n==1
    % All layers sizes are concatenated in a matrix
    layer_size=[input_layer_size'; hidden_layer_size(:,1:n)'; num_labels'];
    
    % nn_params
    % each value is randomly generate
    nn_params=cell(exNum,1);
    paramsSize=zeros(exNum,1);
    
for k=1:exNum
    paramsSize(k)=((input_layer_size(k)+1)*hidden_layer_size(k,1:n)...
                    +num_labels(k)*(hidden_layer_size(k,1:n)+1));
    nn_params{k}=rand(paramsSize(k),1);
end

    
for k=1:exNum
    [J1(k), grad1{k}]=nnCostFunction(nn_params{k}, layer_size(:,k),...
                                n+2, X{k}, y{k}, lambda(k));
    [J2(k), grad2{k}]=nnCostFunction1Hidden(nn_params{k},input_layer_size(k),...
                   hidden_layer_size(k,1:n),num_labels(k),X{k},y{k},lambda(k));
end

fprintf('Hidden layer number: %d\n',n);

% Both results are compared with a tolarence of 0.0000001.
% If some result does not match, it will be printed on screen.
% If not, "OK" will be printed on screen.
ok=1;
for k=1:exNum
    if(abs(J1(k)-J2(k)))>0.0000001
        fprintf('Cost does not match for example %d\n',i);
        ok=0;
    end
    for i=1:paramsSize(k)
        if (abs(grad1{k}(i)-grad2{k}(i))>0.0000001)
            fprintf('Grad %d does not match for example %d\n',i,k);
            ok=0;
        end
    end
end
if ok==1
    fprintf('OK\n');
end
end

if n==2
    % All layers sizes are concatenated in a matrix
    layer_size=[input_layer_size'; hidden_layer_size(:,1:n)'; num_labels'];
    
    % nn_params
    % each value is randomly generate
    nn_params=cell(exNum,1);
    paramsSize=zeros(exNum,1);
    
for k=1:exNum
    paramsSize(k)=((input_layer_size(k)+1)*hidden_layer_size(k,1)...
                    +hidden_layer_size(k,2)*(hidden_layer_size(k,1)+1)...
                    +num_labels(k)*(hidden_layer_size(k,2)+1));
    nn_params{k}=rand(paramsSize(k),1);
end
    
    
for k=1:exNum
    [J1(k), grad1{k}]=nnCostFunction(nn_params{k}, layer_size(:,k),...
                                n+2, X{k}, y{k}, lambda(k));
    [J2(k), grad2{k}]=nnCostFunction2Hidden(nn_params{k},input_layer_size(k),...
                   hidden_layer_size(k,1:n),num_labels(k),X{k},y{k},lambda(k));
end

fprintf('Hidden layer number: %d\n',n);

% Both results are compared with a tolarence of 0.0000001.
% If some result does not match, it will be printed on screen.
% If not, "OK" will be printed on screen.
ok=1;
for k=1:exNum
    if(abs(J1(k)-J2(k)))>0.0000001
        fprintf('Cost does not match for example %d\n',i);
        ok=0;
    end
    for i=1:paramsSize(k)
        if (abs(grad1{k}(i)-grad2{k}(i))>0.0000001)
            fprintf('Grad %d does not match for example %d\n',i,k);
            ok=0;
        end
    end
end
if ok==1
    fprintf('OK\n');
end
end

end










%----------------------------------------------------------------------
%                       Multilayer Cost Function

function [J grad] = nnCostFunction(nn_params, ...
                                   layer_size, ...
                                   layer_num, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, layer_size, layer_num, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network. layer_size contains an 
%   array with the layer size of each layer from input to output layers.
%   layer_num contains the number of layers, counting input, hidden, and
%   output layers.
%

% Reshape nn_params back into the parameters Theta, the weight matrices
% for our neural network

Theta=cell(layer_num-1,1); % Creating a matrix cell array for each Theta

% Initializing reshape range for Theta{1}
params_from=1;
params_to=(layer_size(2) * (layer_size(1) + 1));

% Reshape Theta{1} to Theta{layer_num-1}, the last Theta need to be
% reshaped outside this for loop (problem in params_to assigment for n=layer_num-1)
for n=1:(layer_num-2)
    Theta{n}=reshape(nn_params(params_from : params_to), ...
                 layer_size(n+1), (layer_size(n) + 1), 1);
    params_from=params_to+1;
    params_to=params_to+(layer_size(n+2) * (layer_size(n+1) + 1));
end

% Last Theta reshape
Theta{layer_num-1}=reshape(nn_params(params_from : end), ...
                 layer_size(layer_num), (layer_size(layer_num-1) + 1), 1);

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

Theta_grad=cell(layer_num-1,1); % Theta_grad is also a matrix cell array
for n=1:layer_num-1
Theta_grad{n} = zeros(size(Theta{n}));
end

% --------------------------------  part 1 -------------------------------%

activ=cell(layer_num-1,1); % activ is also a matrix cell array

% pad one to the input X
activ{1} = [ones(m,1) , X];

% input layer to hidden layer for each n-1 to n layer interaction
reg_term=0;
for n=1:layer_num-2
activ{n+1}=[ones(m,1),(sigmoid(activ{n} * Theta{n}'))];
Theta{n}(:,1) = 0; % ensure the first column of theata are zeros
reg_term = lambda / (2 * m) * (sum(sum(Theta{n}.^2)))+reg_term; % reg_term computation
end

% output layer activ matrix is calculated outside of the for loop
% it coul be implemented as activ{layer_num}
out=sigmoid(activ{layer_num-1} * Theta{layer_num-1}');
Theta{layer_num-1}(:,1) = 0;
reg_term = lambda / (2 * m) * (sum(sum(Theta{layer_num-1}.^2)))+reg_term;

% reformat y
rep_y = zeros(m, layer_size(layer_num));
for i = 1:m
   rep_y(i, y(i)) = 1; 
end

% unregularized cost computation
J = sum(sum(-rep_y.* log(out)-(1-rep_y).*log(1-out))) / m;

% regularized cost
J = J + reg_term;

% --------------------------------  part 2 -------------------------------%

% delta is also a matrix cell array. It is important to note that the real
% notation is delta_2, delta_3,..., delta_n, but the implemented cell
% array has indexes 1, 2,..., n-1. Therefore, in this implementation
% delta_2 is reprenseted by delta{1}, for example.

delta=cell(layer_num-1,1);

delta{layer_num-1} = out - rep_y; % delta for the output layer

for i=1:m
    Theta_grad{layer_num-1}=Theta_grad{layer_num-1}...
                +(delta{layer_num-1}(i,:)'*activ{layer_num-1}(i,:));
end

Theta_grad{layer_num-1}=Theta_grad{layer_num-1}./m;
Theta_grad{layer_num-1}=Theta_grad{layer_num-1}+((lambda / m )...
                        .*Theta{layer_num-1});

% delta computation for other deltas
% compute for theta_grad
% accumulate the gradient
for n=layer_num-2:-1:1
    delta{n}=(delta{n+1}*Theta{n+1}).*(activ{n+1} .* (1 - activ{n+1}));
    delta{n}=delta{n}(:,2:end); % get rid of the 1st column
    for i=1:m
        Theta_grad{n}=Theta_grad{n}+(delta{n}(i,:)'*activ{n}(i,:));
    end
    Theta_grad{n}=Theta_grad{n}./m; % gradient is divided by m
    Theta_grad{n}=Theta_grad{n}+((lambda / m ).*Theta{n}); % regulization for gradients
end

% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
for n=1:layer_num-1
Theta_grad{n}=reshape(Theta_grad{n},[],1);
end
grad=cell2mat(Theta_grad);
grad=grad(:);

end










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











%----------------------------------------------------------------------
%                       Two-layer Cost Function

function [J grad] = nnCostFunction2Hidden(nn_params, ...
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
Theta1 = reshape(nn_params(1:hidden_layer_size(1) * (input_layer_size + 1)), ...
                   hidden_layer_size(1), (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size(1) * (input_layer_size + 1))):...
                (hidden_layer_size(2) * (hidden_layer_size(1) + 1))...
                + hidden_layer_size(1) * (input_layer_size + 1)), ...
                 hidden_layer_size(2), (hidden_layer_size(1) + 1));

Theta3 = reshape(nn_params((hidden_layer_size(2) * (hidden_layer_size(1) + 1))...
                +(hidden_layer_size(1) * (input_layer_size + 1))+1:end), ...
                 num_labels, (hidden_layer_size(2) + 1));
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

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
activ1 = sigmoid(tX * Theta1');

% pad one the htheta_1
activ1 = [ones(m,1), activ1];

activ2=sigmoid(activ1*Theta2');

activ2=[ones(m,1), activ2];

% hidden layer to output layer
out = sigmoid(activ2 * Theta3');

% reformat y
rep_y = zeros(m, num_labels);
for i = 1:m
   rep_y(i, y(i)) = 1; 
end

J = sum(sum(-rep_y.* log(out)-(1-rep_y).*log(1-out))) / m;

% ensure the first column of theata are zeros
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta3(:,1) = 0;

reg_term = lambda / (2 * m) * ( sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)) +...
                                sum(sum(Theta3.^2)) );

J = J + reg_term;


% --------------------------------  part 2 -------------------------------%

% Notes:
% y is remapped to rep_y
% activation on output layer is out
% activation on hidden layer is activ

% delta measures how much that a node was "responsible" for the erros in
% the output

delta_4 = out - rep_y;

gz3 = activ2 .* (1 - activ2);
delta_3 = (delta_4 * Theta3) .* gz3;

% compute for theta_grad
% accumulate the gradient
delta_3 = delta_3(:,2:end); % get rid of the 1st column

gz2 = activ1 .* (1 - activ1);
delta_2 = (delta_3 * Theta2) .* gz2;

% compute for theta_grad
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
    cur_alpha = activ1(i,:);
    cur_delta = delta_3(i,:);
    Theta2_grad  = Theta2_grad  + cur_delta' * cur_alpha;
end

Theta2_grad = Theta2_grad ./ m;

% compute for theta2_grad
for i = 1:m
    cur_alpha = activ2(i,:);
    cur_delta = delta_4(i,:);
    Theta3_grad  = Theta3_grad  + cur_delta' * cur_alpha;
end

Theta3_grad = Theta3_grad ./ m;


% --------------------------------  part 3 -------------------------------%

% regulization for gradients
reg_theta1 = (lambda / m ) .* Theta1;
reg_theta2 = (lambda / m ) .* Theta2;
reg_theta3 = (lambda / m ) .* Theta3;

Theta1_grad = Theta1_grad + reg_theta1;
Theta2_grad = Theta2_grad + reg_theta2;
Theta3_grad = Theta3_grad + reg_theta3;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];



end
