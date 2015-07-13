function p = predict(Theta, X, layer_num)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

h=cell(layer_num-1,1);

h{1} = sigmoid([ones(m, 1) X] * Theta{1}');
for n=2:layer_num-1
    h{n}=sigmoid([ones(m, 1) h{n-1}] * Theta{n}');
end

[dummy, p] = max(h{layer_num-1}, [], 2);

% =========================================================================


end
