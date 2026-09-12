function opts = gbcOptions(varargin)
%GBCOPTIONS Default hyper-parameters for the GBC / Implicit Quantile Network.
%
%   opts = GBCOPTIONS() returns the default option struct.
%   opts = GBCOPTIONS('Name',Value,...) overrides individual fields.
%
%   Defaults follow the authors' reference implementation
%   (github.com/VadimSokolov/gbc-surrogate, gbc/iqn.py), which is the
%   authority wherever it and the paper's text differ. See README.md,
%   "Reconciling the paper with the reference code".
%
%   Architecture
%   ------------
%   HiddenSize      Width of f_x, f_tau and f_1 (reference: 256).
%   BottleneckSize  Width of the Tanh layer before the output (reference: 64).
%                   Set 0 to go straight from f_1 to the output head, which is
%                   the architecture Eq. (2) of the paper literally describes.
%   NumCosine       n_h, size of the cosine quantile embedding (reference: 32).
%
%   Loss
%   ----
%   LossWeights     [w1 w2 w3] in Eq. (1). Reference defaults:
%                     [0.3 0.3 0.4] smooth / heteroskedastic responses
%                     [0.1 0.2 0.7] "quantile-dominant", for jump processes
%
%   Optimisation
%   ------------
%   MaxEpochs       Gradient steps when full batch (reference default: 3000;
%                   the motorcycle table uses 5000).
%   MiniBatchSize   Inf for full-batch training, which is what the reference
%                   does. A finite value switches to shuffled mini-batches.
%   Accelerate      Wrap the loss in dlaccelerate, which caches the traced
%                   computation graph instead of rebuilding it every step.
%                   Profiling shows the per-step cost is dominated by a FIXED
%                   ~5-8 ms of tracing overhead that does not shrink with n
%                   and is identical on CPU and GPU - on GPU it is the whole
%                   step time up to n = 20000. This targets exactly that.
%                   ON by default: measured 2.7x on CPU at n = 106 and 2.3x
%                   on GPU at n = 20000, computing the same thing - the loss
%                   is branch-free in tau and tau is passed as a traced
%                   dlarray, so no cached trace can freeze a stale value.
%                   Set false to measure the difference; test_gbc checks that
%                   accelerated and plain training agree.
%   TauPerExample   false (reference): ONE tau ~ U[0,1] per gradient step,
%                   shared by every example in the batch.
%                   true: an independent tau per example, which covers the
%                   quantile curve faster per step but is not what the
%                   reference does.
%   InitialLR       Adam learning rate (reference: 1e-3).
%   MinLR           Cosine-annealing floor. [] means 0.01*InitialLR, matching
%                   the reference's eta_min.
%   WeightDecay     Adam weight decay, applied exactly as torch.optim.Adam
%                   does it: wd*theta added to the gradient of every
%                   parameter, biases included (reference: 1e-4).
%   GradientDecay   Adam beta1.
%   SqGradDecay     Adam beta2.
%
%   Data handling
%   -------------
%   Standardize     Z-score inputs and response (the reference does this).
%
%   Surrogate sample set (calibration)
%   ----------------------------------
%   NumSamples      Size of the fixed tau sample set the fitted gbcModel
%                   carries (default 500). An outer MCMC sampler indexes into
%                   it to draw surrogate uncertainty reproducibly; see
%                   GBCMODEL. Raise it if the chain should see more than 500
%                   distinct surrogate realisations.
%   SampleSeed      Seed for that sample set. [] inherits Seed, so ensemble
%                   members get different sample sets.
%
%   Run control
%   -----------
%   ExecutionEnvironment  'auto' | 'cpu' | 'gpu'.
%   Verbose         Print progress.
%   VerboseFreq     Print every N epochs.
%   ValidationData  {Xval,Yval} cell array, or [] for none.
%   Seed            RNG seed, or [] to leave the global stream alone.
%
%   See also GBCTRAIN, GBCENSEMBLE, GBCPREDICT, GBCMODEL, GBCMETRICS.

opts = struct( ...
    'HiddenSize',           256, ...
    'BottleneckSize',       64, ...
    'NumCosine',            32, ...
    'LossWeights',          [0.3 0.3 0.4], ...
    'MaxEpochs',            3000, ...
    'MiniBatchSize',        Inf, ...
    'TauPerExample',        false, ...
    'Accelerate',           true, ...
    'InitialLR',            1e-3, ...
    'MinLR',                [], ...
    'WeightDecay',          1e-4, ...
    'GradientDecay',        0.9, ...
    'SqGradDecay',          0.999, ...
    'Standardize',          true, ...
    'NumSamples',           500, ...
    'SampleSeed',           [], ...
    'ExecutionEnvironment', 'auto', ...
    'Verbose',              true, ...
    'VerboseFreq',          250, ...
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
validateattributes(opts.BottleneckSize,{'numeric'},{'scalar','nonnegative','integer'});
validateattributes(opts.WeightDecay,{'numeric'},{'scalar','nonnegative'});
validateattributes(opts.NumSamples,{'numeric'},{'scalar','positive','integer'});

if isempty(opts.MinLR)
    opts.MinLR = 0.01 * opts.InitialLR;    % torch CosineAnnealingLR eta_min
end
end
