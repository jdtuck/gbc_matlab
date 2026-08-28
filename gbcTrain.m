function model = gbcTrain(X, Y, opts)
%GBCTRAIN Train a Generative Bayesian Computation surrogate (IQN).
%
%   model = GBCTRAIN(X,Y) trains an Implicit Quantile Network on the
%   input/output pairs of an expensive computer experiment, following
%   Algorithm 1 of Polson & Sokolov (2026) and the authors' reference
%   implementation (gbc/iqn.py).
%
%   model = GBCTRAIN(X,Y,opts) uses the option struct from GBCOPTIONS.
%
%   Inputs
%     X : n-by-d matrix of design points.
%     Y : n-by-1 vector of (noisy) simulator responses.
%
%   Output
%     model : struct consumable by GBCPREDICT / GBCSAMPLE, containing the
%             learned parameters, the standardisation constants and a
%             training history.
%
%   By default this reproduces the reference recipe: full-batch Adam with
%   weight decay 1e-4, cosine annealing from 1e-3 down to 1e-5, and a single
%   tau ~ U[0,1] drawn per gradient step and shared across the batch. Set
%   opts.MiniBatchSize and opts.TauPerExample to depart from that.
%
%   Example
%     X = rand(2000,10);
%     Y = 10*sin(pi*X(:,1).*X(:,2)) + 20*(X(:,3)-0.5).^2 + 10*X(:,4) + ...
%         5*X(:,5) + randn(2000,1);
%     model = gbcTrain(X,Y,gbcOptions('MaxEpochs',3000));
%
%   See also GBCOPTIONS, GBCENSEMBLE, GBCPREDICT, GBCSAMPLE, GBCMETRICS.

arguments
    X (:,:) double
    Y (:,1) double
    opts struct = gbcOptions()
end

if size(X,1) ~= numel(Y)
    error('gbcTrain:SizeMismatch','X has %d rows but Y has %d elements.', ...
          size(X,1), numel(Y));
end
if ~isempty(opts.Seed)
    rng(opts.Seed);
end

[n, d] = size(X);
w  = opts.LossWeights(:).';
nh = opts.NumCosine;

% ---------------------------------------------------------------- scaling
if opts.Standardize
    % std normalised by N, not N-1: numpy's default, which is what the
    % reference uses (X_np.std(0) + 1e-8).
    muX = mean(X,1);
    sdX = std(X,1,1) + 1e-8;
    muY = mean(Y);
    sdY = std(Y,1) + 1e-8;
else
    muX = zeros(1,d); sdX = ones(1,d); muY = 0; sdY = 1;
end
Xs = (X - muX) ./ sdX;
Ys = (Y - muY) ./ sdY;

% ---------------------------------------------------------------- device
useGPU = shouldUseGPU(opts.ExecutionEnvironment);

Xall = single(Xs.');            % d-by-n
Yall = single(Ys.');            % 1-by-n
if useGPU
    Xall = gpuArray(Xall);
    Yall = gpuArray(Yall);
end

% ---------------------------------------------------------------- params
params = gbcInit(d, opts);
if useGPU
    params = dlupdate(@gpuArray, params);
end

avgG   = [];
avgSqG = [];

if isinf(opts.MiniBatchSize)
    batch = n;                  % full batch, as in the reference
else
    batch = min(opts.MiniBatchSize, n);
end
nBatch    = max(1, ceil(n/batch));
totalIter = opts.MaxEpochs * nBatch;

% History is recorded only on checkpoint epochs, so these are sized by the
% number of checkpoints rather than by MaxEpochs.
nCheck   = numel(unique([1, opts.VerboseFreq:opts.VerboseFreq:opts.MaxEpochs, ...
                         opts.MaxEpochs]));
lossLog  = nan(nCheck,4);
lrLog    = nan(nCheck,1);
valLog   = nan(nCheck,1);
epochLog = nan(nCheck,1);

hasVal = ~isempty(opts.ValidationData);
if hasVal
    Xval = opts.ValidationData{1};
    Yval = opts.ValidationData{2};
    tauGridVal = (1:99)/100;
end

if opts.Verbose
    fprintf('GBC-IQN: n=%d d=%d width=%d bottleneck=%d steps=%d batch=%s tau=%s dev=%s\n', ...
        n, d, opts.HiddenSize, opts.BottleneckSize, opts.MaxEpochs, ...
        ternary(isinf(opts.MiniBatchSize),'full',num2str(batch)), ...
        ternary(opts.TauPerExample,'per-example','per-step'), ...
        ternary(useGPU,'gpu','cpu'));
    fprintf('%8s %12s %10s %10s %10s %10s\n', ...
        'epoch','loss','anchor','order','pinball','lr');
end

% In full-batch mode every step sees exactly the same X and Y, so building
% the dlarray inside the loop would rebuild an identical object thousands of
% times. Hoist it.
fullBatch = (nBatch == 1);
if fullBatch
    XbFixed = dlarray(Xall, 'CB');
    YbFixed = Yall;
end

iter = 0;
nLogged = 0;
t0 = tic;
for epoch = 1:opts.MaxEpochs

    atCheckpoint = mod(epoch, opts.VerboseFreq) == 0 || epoch == 1 || ...
                   epoch == opts.MaxEpochs;

    if nBatch > 1
        idx = randperm(n);
    else
        idx = [];               % full batch: no shuffling needed
    end
    epochParts = zeros(1,3);
    epochLoss  = 0;

    for b = 1:nBatch
        iter = iter + 1;

        if fullBatch
            Xb = XbFixed;  Yb = YbFixed;  nb = n;
        else
            sel = idx((b-1)*batch + 1 : min(b*batch, n));
            nb  = numel(sel);
            Xb  = dlarray(Xall(:,sel), 'CB');
            Yb  = Yall(:,sel);
        end

        % Reference: one tau per gradient step, shared across the batch.
        % Optional: an independent tau per example.
        if opts.TauPerExample
            tau = rand(1, nb, 'single');
        else
            tau = repmat(rand(1,1,'single'), 1, nb);
        end
        if useGPU, tau = gpuArray(tau); end
        Phi = dlarray(quantileEmbedding(tau, nh), 'CB');

        % Only ask for the loss value and its breakdown on epochs we actually
        % record. Each extractdata forces a copy (and a device sync on GPU),
        % and at one gradient step per epoch that cost lands on every step.
        if atCheckpoint
            [lossVal, grads, parts] = dlfeval(@gbcLoss, params, Xb, Phi, ...
                                              tau, Yb, w, 0);
            epochLoss  = epochLoss  + gather(double(extractdata(lossVal)));
            epochParts = epochParts + parts;
        else
            [~, grads] = dlfeval(@gbcLoss, params, Xb, Phi, tau, Yb, w, 0);
        end

        % Adam weight decay exactly as torch.optim.Adam applies it: added to
        % the gradient of every parameter, biases included. (This is the
        % coupled L2 form, not AdamW's decoupled decay.)
        if opts.WeightDecay > 0
            grads = dlupdate(@(g,p) g + opts.WeightDecay*p, grads, params);
        end

        lr = cosineAnneal(opts.InitialLR, opts.MinLR, iter, totalIter);

        [params, avgG, avgSqG] = adamupdate(params, grads, avgG, avgSqG, ...
            iter, lr, opts.GradientDecay, opts.SqGradDecay);
    end

    if atCheckpoint
        nLogged = nLogged + 1;
        lossLog(nLogged,:) = [epochLoss, epochParts] / nBatch;
        lrLog(nLogged)     = lr;
        epochLog(nLogged)  = epoch; %#ok<AGROW>
    end

    if hasVal && atCheckpoint
        tmp = gbcModel(params, opts, muX, sdX, muY, sdY, d, [], []);
        valLog(nLogged) = gbcCRPS(gbcPredict(tmp, Xval, tauGridVal), Yval);
    end

    if opts.Verbose && atCheckpoint
        if hasVal
            fprintf('%8d %12.5f %10.4f %10.4f %10.4f %10.2e   valCRPS %.4f\n', ...
                epoch, lossLog(nLogged,1), lossLog(nLogged,2), ...
                lossLog(nLogged,3), lossLog(nLogged,4), lr, valLog(nLogged));
        else
            fprintf('%8d %12.5f %10.4f %10.4f %10.4f %10.2e\n', ...
                epoch, lossLog(nLogged,1), lossLog(nLogged,2), ...
                lossLog(nLogged,3), lossLog(nLogged,4), lr);
        end
    end
end

keep = 1:nLogged;
history.epoch     = epochLog(keep);
history.loss      = lossLog(keep,1);
history.anchor    = lossLog(keep,2);
history.ordering  = lossLog(keep,3);
history.pinball   = lossLog(keep,4);
history.lr        = lrLog(keep);
history.valCRPS   = valLog(keep);
history.trainTime = toc(t0);

model = gbcModel(params, opts, muX, sdX, muY, sdY, d, history, n);

if opts.Verbose
    fprintf('Done in %.1f s.\n', history.trainTime);
end

end

% =========================================================================

function lr = cosineAnneal(lrMax, lrMin, iter, total)
%COSINEANNEAL Loshchilov & Hutter (2017) single-cycle cosine schedule, in the
%   form torch.optim.lr_scheduler.CosineAnnealingLR uses.
lr = lrMin + 0.5*(lrMax - lrMin)*(1 + cos(pi * (iter-1) / max(1,total-1)));
end

function tf = shouldUseGPU(mode)
switch lower(mode)
    case 'gpu'
        tf = true;
    case 'cpu'
        tf = false;
    otherwise
        tf = false;
        try
            tf = gpuDeviceCount("available") > 0; %#ok<GPUDEV>
        catch
            try
                tf = gpuDeviceCount > 0;
            catch
                tf = false;
            end
        end
end
end

function out = ternary(c,a,b)
if c, out = a; else, out = b; end
end
