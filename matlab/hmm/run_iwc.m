%% Isolated Word Recognition using HMM

% Reference: isolated-word speech recognition using Hidden Markov Model 
% by Håkon Sandsmark , Dec 2010

clear
clc

[audio_signals, word_labels] = load_audio_from_folder('database');
[unique_word_labels, ~, indices] = unique(word_labels);

unique_words = length(unique_word_labels);
wordHMMs = word_hmm(1,unique_words);

%% configure the training set and testing set
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


%%
% sampling frequency
fs = 8000;

display(sprintf('Training ...'));

% go through each unique words
for i = 1:length(unique_word_labels)
    
    % initialize
    w = word_hmm(); 
    w.name = char(unique_word_labels(i));
    display(sprintf('Training on ''%s''...', w.name));

    % load the training samples
    training_samples = training_set{i};

    % make sure the input for each record is the same length
    % and concatenate them
    speech = trimSig(training_samples);
    sample_ele = numel(speech);
    speech = reshape(speech,sample_ele,1); % reshape in column major
    
    % feature extraction: Mel Freq Cepstrum Coefficients
    observations = feature_mfcc(speech,fs);
    
    % initialize hmm
    w.prior = normalise(rand(w.N, 1));
    w.A     = mk_stochastic(rand(w.N));
    % All states start out with the empirical (diagonal) covariance
    w.Sigma = repmat(diag(diag(cov(observations'))), [1 1 w.N]);
    % Initialize each mean to a random data point
    ind = randperm(size(observations, 2));
    w.mu = observations(:, ind(1:w.N));
    
    % EM / Bawm-Welch Algorithm: iterative training
    for iter = 1:20
        w = train_hmm(observations, w);
    end
    
    % save the results
    wordHMMs(i) = w;
end

display(sprintf('Finish Training\n'));


%%  test hmm

display(sprintf('Testing ...\n'));

% test for each word
for i = 1 : unique_words
    
    samples    = testing_set{i};
    sample_num = length(samples);
    
    target     = cell2mat(unique_word_labels(i));
    
    display(sprintf('Target word : %s', target));
    
    %load test samples for each word
    for sid = 1 : sample_num
        
        % convert cell to mat
        current_speech = cell2mat(samples(sid));
        
        %log likelihood
        ll = zeros(size(wordHMMs));

        % iterate throught each hmm, and find the best loglikelihood
        for hid = 1 : unique_words
            % load the sample and trained HMM
            ll(1, hid) = test_hmm(current_speech, wordHMMs(1,hid), fs);
        end
        
        % find the max lll
        [maxll, maxInd] = max(ll);
        result = cell2mat(unique_word_labels(maxInd));
        display(sprintf('Recognized as %s', result));
    end
    

    display(sprintf('------------------------'));
end



