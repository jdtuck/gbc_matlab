function opts = gbcOptions(varargin)
%GBCOPTIONS Default hyper-parameters for the GBC / Implicit Quantile Network.
%
%   opts = GBCOPTIONS() returns the default option struct.
%   opts = GBCOPTIONS('Name',Value,...) overrides individual fields.
%
%   Defaults follow Polson & Sokolov (2026), "Generative Bayesian Computation
%   as a Scalable Alternative to Gaussian Process Surrogates", arXiv:2602.21408.
%
%   Fields
%   ------
%   HiddenSize     Width of f_x, f_tau and f_1 (paper: 256).
%   NumCosine      n_h, size of the cosine quantile embedding (paper: 32).
%   LossWeights    [w1 w2 w3] in Eq. (1). Paper defaults:
%                    [0.3 0.3 0.4] smooth / heteroskedastic responses
%                    [0.1 0.2 0.7] "quantile-dominant", for jump processes
%   MaxEpochs      Paper uses 3,000-8,000 depending on dataset size.
%   MiniBatchSize  Not stated in the paper; 256 is a reasonable default.
%   InitialLR      Adam learning rate (paper: 1e-3).
%   MinLR          Floor of the cosine-annealing schedule.
%   GradientDecay  Adam beta1.
%   SqGradDecay    Adam beta2.
%   L2Regularization  Weight decay applied to weight matrices (0 = off).
%   Standardize    Z-score inputs and response (implementation choice; the
%                  paper does not specify a normalisation protocol).
%   ExecutionEnvironment  'auto' | 'cpu' | 'gpu'.
%   Verbose        Print progress.
%   VerboseFreq    Print every N epochs.
%   ValidationData {Xval,Yval} cell array, or [] for none.
%   Seed           RNG seed, or [] to leave the global stream alone.
%
%   See also GBCTRAIN, GBCPREDICT, GBCMETRICS.

opts = struct( ...
    'HiddenSize',           256, ...
    'NumCosine',            32, ...
    'LossWeights',          [0.3 0.3 0.4], ...
    'MaxEpochs',            3000, ...
    'MiniBatchSize',        256, ...
    'InitialLR',            1e-3, ...
    'MinLR',                0, ...
    'GradientDecay',        0.9, ...
    'SqGradDecay',          0.999, ...
    'L2Regularization',     0, ...
    'Standardize',          true, ...
    'ExecutionEnvironment', 'auto', ...
    'Verbose',              true, ...
    'VerboseFreq',          100, ...
    'ValidationData',       [], ...
    'Seed',                 []);

if mod(numel(varargin),2) ~= 0
    error('gbcOptions:BadArgs','Options must be supplied as name/value pairs.');
end
for k = 1:2:numel(varargin)
    name = varargin{k};
    if ~isfield(opts,name)
        error('gbcOptions:UnknownOption','Unknown option "%s".',name);
    end
    opts.(name) = varargin{k+1};
end

validateattributes(opts.LossWeights,{'numeric'},{'vector','numel',3,'nonnegative'});
validateattributes(opts.HiddenSize,{'numeric'},{'scalar','positive','integer'});
validateattributes(opts.NumCosine,{'numeric'},{'scalar','positive','integer'});
end
