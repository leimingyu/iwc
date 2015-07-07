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
for n=2:layer_num-1
activ{n}=[ones(m,1),(sigmoid(activ{n-1} * Theta{n-1}'))];
end

% output layer activ matrix is calculated outside of the for loop
% it coul be implemented as activ{layer_num}
out=sigmoid(activ{layer_num-1} * Theta{layer_num-1}');

% reformat y
rep_y = zeros(m, layer_size(layer_num));
for i = 1:m
   rep_y(i, y(i)) = 1; 
end

% unregularized cost computation
J = sum(sum(-rep_y.* log(out)-(1-rep_y).*log(1-out))) / m;

% ensure the first column of theata are zeros
for n=1:layer_num-1
Theta{n}(:,1) = 0;
end

% reg_term computation
reg_term=0;
for n=1:layer_num-1
reg_term = lambda / (2 * m) * (sum(sum(Theta{n}.^2)))+reg_term;
end

% regularized cost
J = J + reg_term;

% --------------------------------  part 2 -------------------------------%

% delta is also a matrix cell array. It is important to note that the real
% notation is delta_2, delta_3,..., delta_n, but the implemented cell
% array has indexes 1, 2,..., n-1. Therefore, in this implementation
% delta_2 is reprenseted by delta{1}, for example.

delta=cell(layer_num-1,1);

delta{layer_num-1} = out - rep_y; % delta for the output layer

% delta computation for other deltas
for n=layer_num-2:-1:1
    delta{n}=(delta{n+1}*Theta{n+1}).*(activ{n+1} .* (1 - activ{n+1}));
    delta{n}=delta{n}(:,2:end); % get rid of the 1st column
end


% compute for theta1_grad
% accumulate the gradient
for i=1:m
    for n=1:layer_num-1
        Theta_grad{n}=Theta_grad{n}+(delta{n}(i,:)'*activ{n}(i,:));
    end
end

% gradient is divided by m
for n=1:layer_num-1
    Theta_grad{n}=Theta_grad{n}./m;
end

% --------------------------------  part 3 -------------------------------%

% regulization for gradients
for n=1:layer_num-1
Theta_grad{n}=Theta_grad{n}+((lambda / m ).*Theta{n});
end

% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
for n=1:layer_num-1
Theta_grad{n}=reshape(Theta_grad{n},[],1);
end
grad=cell2mat(Theta_grad);

end

%----------------------------------------------------------------------
%                       Test for Multilayer Cost Function
%
%   The multilayer version is set to 3 layers and compared to a three layer
%   version.

layer_num=3; % Number of layers

exNum=10; % Number of random examples
rangeInt=[5 100]; % Range for radom integer numbers vary

m=randi(rangeInt,exNum,1); % Random number of samples
input_layer_size=randi(rangeInt,exNum,1); % Random size for the input layer
hidden_layer_size=randi(rangeInt,exNum,1); % Random size for the hidden layer
num_labels=randi(rangeInt,exNum,1); % Random number of labels

% nn_params, X, and y are matrix cell arrays of exNum size
% each value is randomly generate
nn_params=cell(exNum,1);
paramsSize=zeros(exNum,1);
for k=1:exNum
    paramsSize(k)=((input_layer_size(k)+1)*hidden_layer_size(k)+num_labels(k)*(hidden_layer_size(k)+1));
    nn_params{k}=rand(paramsSize(k),1);
end

X=cell(exNum,1);
for k=1:exNum
    X{k}=rand(m(k),input_layer_size(k));
end

y=cell(exNum,1);
for k=1:exNum
    y{k}=randi([1 num_labels(k)],m(k),1);
end

lambda=rand(exNum,1); % Random lambdas

% All layers sizes are concatenated in a matrix
layer_size=[input_layer_size'; hidden_layer_size'; num_labels'];

% Arrays for both costs and cell for both grads are created
J1=zeros(exNum,1);
J2=zeros(exNum,1);
grad1=cell(exNum,1);
grad2=cell(exNum,1);


% Both cost functions are run using the same random arguments.
% nnCostFunction is the provided two-layer function and nnCostFunction2 is
% the implemented multilayer approach.
for k=1:exNum
[J1(k), grad1{k}]=nnCostFunction(nn_params{k},input_layer_size(k),...
                hidden_layer_size(k),num_labels(k),X{k},y{k},lambda(k));
[J2(k), grad2{k}]=nnCostFunction2(nn_params{k}, layer_size(:,k),...
                                layer_num, X{k}, y{k}, lambda(k));
end


% Both results are compared with a tolarence of 0.0000001.
% If some result does not match, it will be printed on screen.
% If not, "OK" will be printed on screen.
ok=1;
for k=1:exNum
    if(abs(J1(k)-J2(k)))>0.0000001
        fpitf('Cost does not match for example %d\n',i);
        ok=0;
    end
    for i=1:paramsSize(k)
        if (abs(grad1{k}(i)-grad2{k}(i))>0.0000001)
            fpitf('Grad %d does not match for example %d\n',i,k);
            ok=0;
        end
    end
end
if ok==1
    fprintf('OK\n');
end
