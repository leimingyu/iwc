function outputsignal = trimSig(org_word)
% Find the min length of input signal
% truncate each signal to the same min length
%  org_word is cell
% outputsignal is matrix


%  N is the number of samples
N = size(org_word,2);
lenN = zeros(1,N);
signal = cell(1,N);

for i =  1: N % number of input samples
    lenN(i)  = length(cell2mat((org_word(1,i)))); % column major
end

[minlen, ~] = min(lenN);

% try to vectorize this loop
for i =  1: N
    m = cell2mat(org_word(1,i));
    m = m(1:minlen);
    signal(1,i) = {m};
end


outputsignal = cell2mat(signal);  
    
end




    