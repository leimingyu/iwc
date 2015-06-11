function [dct_mat] = dctm(cep, N)
% matrix of dct coefficients
%
% input:
%       cep is the number of cepstral coefficients
%       N is the number of filter banks
% author: leiming yu, 2015

dct_mat = zeros(cep, N);

wk = sqrt(2/N);
step = 1:2:(2 * N - 1);

for k = 1:cep
    
    dct_mat(k,:) = wk * cos( (k-1) * pi * step / (2 * N));
    
end

end
% EOF