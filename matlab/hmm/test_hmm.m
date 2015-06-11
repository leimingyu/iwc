function [llh] = test_hmm(current_speech, hmm_prop, fs)
% load HMM 
N       = hmm_prop.N;
prior   = hmm_prop.prior;
A       = hmm_prop.A;
mu      = hmm_prop.mu;
Sigma   = hmm_prop.Sigma;

% extract features
observations = feature_mfcc(current_speech, fs);
% frames of speech 
T = size(observations, 2);

% observations
B = zeros(N, size(observations, 2));            
for s = 1:N
    B(s, :) = mvnpdf(observations', mu(:, s)', Sigma(:, :, s));
end

log_likelihood = 0;

alpha = zeros(N, T);

% go throught each frame, update the forward prob, alpha
for t = 1:T
    if t == 1
        alpha(:, t) = B(:, t) .* prior;
    else
        alpha(:, t) = B(:, t) .* (A' * alpha(:, t - 1));
    end
    
    % Scaling
    alpha_sum      = sum(alpha(:, t));
    alpha(:, t)    = alpha(:, t) ./ alpha_sum;
    log_likelihood = log_likelihood + log(alpha_sum);
end

llh = log_likelihood;

end
%EOF