%% Isolated Word Recognition using Neural Networks
% Author: Leiming Yu, ylm@ece.neu.edu
%         Northeastern University, Boston, MA, USA
%
% The NN code is referred from Machine Learning Course (Andrew Ng), Cousera

clear
clc

[audio_signals, word_labels] = load_audio_from_folder('database');
[unique_word_labels, ~, indices] = unique(word_labels);

unique_words = length(unique_word_labels);
% wordHMMs = word_hmm(1,unique_words);

% configure the training set and testing set
training_set = cell(1, unique_words);
testing_set  = cell(1, unique_words);

for i = 1:unique_words
    
    % concatinate the speech samples
    org_word = audio_signals(indices==i);
    
    % separate to training / testing sets, with ratio 0.8
    % fixme: add cross validation sets
    ind = randperm(length(org_word));
    training_sample_num = ceil(length(org_word) * 0.8);

    training_samples = org_word( ind(1:training_sample_num) );
    testing_samples  = org_word( ind(training_sample_num + 1 : end) );
    
    training_set{i} = training_samples;
    testing_set{i}  = testing_samples;

end


fs = 8000; % 8 KHz as sampling freq

% training NN
fprintf('\nTraining Neural Network... \n')

% configure
input_layer_size  = 195;           % 195 MFCCs ( each sample )
hidden_layer_size = 31;            % 31 hidden uinits
num_labels        = unique_words;  % 7 unique words

trained_nn = cell(1, unique_words);

% go through each unique words
for id = 1:length(unique_word_labels)
    
    fprintf('Training on ''%s''...\n', char(unique_word_labels(id)) );
    % load samples
    training_samples = training_set{id};
    
    % make sure the input for each record is the same length
    % fixme: how to make sure generate the same feature dimensions for NN
    % row : data
    % col : sample id
    speech = trimSig(training_samples);
    
    % feature array for all the samples
    % 13 coef/frame * 15 (frames) = 195(coef. per speech sample)
    % row : sample id
    % col : features
    X = zeros(size(speech, 2), 195);
    

    % extract feature for each speech sample
    for sid = 1 : size(speech, 2)  
        observations = feature_mfcc(speech(:, sid), fs);
        frame_num =  size(observations,2);
        if frame_num < 15
            fprintf('not enough samples');
            exit();
        end
        midpoint = ceil(frame_num / 2);
        

        % 3 frames at the beginning,
        % 9 frames in the middle
        % 3 frames at the end
        fmt_obs = [observations(:,(1:3)), ...
            observations(:, (midpoint-4: midpoint+4)), ...
            observations(:, (end-2:end))];
        
        % reshape: the output is in column, transpose it to row
        X(sid, :) = reshape(fmt_obs, numel(fmt_obs),1)';
    end
    
    y = zeros(size(X,1), 1);
    y(:) = id;
    
    % padded 1s for the theta
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    
    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

    % training iterations
    options = optimset('MaxIter', 50);
    
    % regularization
    lambda = 0.1;
    
    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
        input_layer_size, ...
        hidden_layer_size, ...
        num_labels, X, y, lambda);
    
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    
    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels, (hidden_layer_size + 1));
    
    % save Theta1 and Theta2 as the NN parameters for current word

    trained_nn{id} = {{Theta1} , {Theta2}};
    
end


% Test

fprintf('\nPredicting using NN ...\n');

% read testing set for each word
for wid = 1 : length(unique_word_labels)
    
    fprintf('Target word : ''%s''', char(unique_word_labels(id)) );
    
    % load Xtest (test samples)
    testing_samples = testing_set{wid};
    
    % extract features
    speech = trimSig(testing_samples);
    Xtest = zeros(size(speech, 2), 195);
    for sid = 1 : size(speech, 2)
        observations = feature_mfcc(speech(:, sid), fs);
        frame_num =  size(observations,2);
        if frame_num < 15
            fprintf('not enough samples');
            exit();
        end
        midpoint = ceil(frame_num / 2);
        
        fmt_obs = [observations(:,(1:3)), ...
            observations(:, (midpoint-4: midpoint+4)), ...
            observations(:, (end-2:end))];
        
        Xtest(sid, :) = reshape(fmt_obs, numel(fmt_obs),1)';
    end
    
    % load theta(s)
    thetas = trained_nn{wid};
    Theta1 = thetas{1,1}{1,1};
    Theta2 = thetas{1,2}{1,1};
    
    % prediction
    pred = predict(Theta1, Theta2, Xtest);
    
    ytest = zeros(size(speech, 2),1);
    ytest(:) = wid;
    
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);
    
end

