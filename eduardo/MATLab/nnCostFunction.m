
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
