
%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
layer_size=[input_layer_size;input_layer_size;num_labels];
layer_num=size(layer_size,1);
                          % (note that we have mapped "0" to label 10)
lambda=0.03;
learning_rate=0.005;
momentum=0.9;
batch_size=100;
epochs=50;

op=2;

results=cell(2,1);
for i=1:2
    results{i}=struct('time',[],'accuracy',[]);
    results{i}.time=zeros(10,1);
    results{i}.accuracy=zeros(10,1);
end

for op=1:2
    for e=1:4
epochs=25*e;
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
m = size(X, 1);


%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

intial_Theta=cell(layer_num-1);
for n=1:layer_num-1
    initial_Theta{n}=randInitializeWeights(layer_size(n),layer_size(n+1));
end

initial_nn_params=[];
% Unroll parameters
for n=1:layer_num-1
    initial_Theta{n}=reshape(initial_Theta{n},[],1);
    initial_nn_params=[initial_nn_params;initial_Theta{n}];
end

%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

Theta=cell(layer_num-1,1);

if op==1
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', epochs);

tic;
X=scaling(X);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   layer_size, ...
                                   layer_num, ...
                                   X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
results{op}.time(e)=toc;
fprintf('\nTraining Time: %f\n',results{op}.time(e));
end

if op==2
costFunction = @(p,in,out) nnCostFunction(p, ...
                                   layer_size, ...
                                   layer_num, ...
                                   in, out, lambda);
tic;

X=scaling(X);

[nn_params, cost] = miniBatch(costFunction,initial_nn_params,X,y,...
                        learning_rate,momentum,batch_size,epochs);
results{op}.time(e)=toc;
fprintf('\nTraining Time: %f\n',results{op}.time(e));
end

% Initializing reshape range
params_to=0;

% Reshape Theta{1} to Theta{layer_num-1}
for n=1:(layer_num-1)
    params_from=params_to+1;    
    params_to=params_to+(layer_size(n+1) * (layer_size(n) + 1));
    Theta{n}=reshape(nn_params(params_from : params_to), ...
                 layer_size(n+1), (layer_size(n) + 1));
end
fprintf('Program paused. Press enter to continue.\n');


%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta{1}(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.


pred = predict(Theta, X, layer_num);
fprintf('\nTraining Set Accuracy for model: %f\n', mean(double(pred == y)) * 100);
results{op}.accuracy(e)= mean(double(pred == y)) * 100;
    end
end
