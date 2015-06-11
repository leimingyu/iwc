classdef word_hmm 
    properties
        N     =  3; % number of states
        A     = []; % NxN transition probability matrix
        prior = []; % Nx1 initial state distribution vector
        mu    = []; % DxN mean vector (D = number of features)
        Sigma = []; % DxDxN covariance matrix
        name = '';
    end
    
    methods
       
        function obj = word_hmm(varargin)
            if nargin > 0 % Allow nargin == 0 syntax
                obj = repmat(obj,[varargin{:}]);
            end
        end

    end
end
