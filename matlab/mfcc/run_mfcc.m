%% MFCC demo
% refer to : 
% http://www.mathworks.com/matlabcentral/fileexchange/32849-htk-mfcc-matlab
% /content/mfcc/mfcc.m
%

clc;
clear;

% step 1
% read speech samples, sampling rate and precision from file
[ speech, fs] = audioread( 'apple01.wav' );
soundsc(speech, fs);
% soundsc(speech, 1.5 * fs);

% normalize the sound
speech = speech / max(abs(speech));

% step 2
% preemphasis filter
alpha = 0.97;
speech = filter( [1 -alpha], 1, speech ); % fvtool( [1 -alpha], 1 );
soundsc(speech, fs);

% step 3
% truncate into (overlapping) frames/windows
% fixme: change according to the time duration
framesize = 256;
overlap = 128;
frames = buffer(speech, framesize, overlap);
col_frames = size(frames, 2);

% step 4
% hanning window or hamming
window_coef = hann(framesize);
w = repmat(window_coef, 1, col_frames);
frames =  frames .* w;

% step 5
% fft and compute the power
nfft = 2^nextpow2(framesize); % round the upper bound of power of 2
MAG = abs( fft(frames,nfft,1) ); 

% step 6
% produce triangular (filterbank) coefficients with uniformly spaced filters on mel scale
M = 26;                 % number of filterbank channels 
C = 13;                 % number of cepstral coefficients 
K = nfft/2+1;           % length of the unique part of the FFT 

LF = 300;               % lower frequency limit (Hz)
HF = 3700;              % upper frequency limit (Hz)
R = [LF HF];            % specifies frequency limits (Hz)

hz2mel = @(hz)(1127*log(1+hz/700));     % Hertz to mel warping function
mel2hz = @(mel)(700*exp(mel/1127)-700); % mel to Hertz warping function

H = trifbank( M, K, R, fs, hz2mel, mel2hz); % size of H is M x K 

% step 6
% Filterbank application to unique part of the magnitude spectrum
FBE = H * MAG(1:K,:); % FBE( FBE<1.0 ) = 1.0; % apply mel floor

% step 7 
% generate the DCT coefficients
DCT = dctm( C, M );

% step 8 
% mulitply the coef. with log (FBE)
CC =  DCT * log( FBE );

% step 9
% Cepstral lifter computation
L = 22;                 % cepstral sine lifter parameter

% Cepstral lifter routine (see Eq. (5.12) on p.75 of [htk book])
% high order cepstra are small, leading to high variance
ceplifter = @( N, L )( 1 + 0.5 * L * sin( pi * [0:N-1] / L) );
lifter = ceplifter( C, L );

% step 10
% Cepstral liftering gives liftered cepstral coefficients
CC = diag( lifter ) * CC; % ~ HTK's MFCCs



