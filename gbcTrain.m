function model = gbcTrain(X, Y, opts)
%GBCTRAIN Train a Generative Bayesian Computation surrogate (IQN).
%
%   model = GBCTRAIN(X,Y) trains an Implicit Quantile Network on the
%   input/output pairs of an expensive computer experiment, following
%   Algorithm 1 of Polson & Sokolov (2026), arXiv:2602.21408.
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
%   Training draws a single tau ~ U[0,1] per example per mini-batch, so over
%   many epochs the network sees the whole quantile curve without ever having
%   to store it. Optimisation is Adam with a cosine-annealed learning rate.
%
%   Example
%     X = rand(2000,10);
%     Y = 10*sin(pi*X(:,1).*X(:,2)) + 20*(X(:,3)-0.5).^2 + 10*X(:,4) + ...
%         5*X(:,5) + randn(2000,1);
%     model = gbcTrain(X,Y,gbcOptions('MaxEpochs',1000));
%
%   See also GBCOPTIONS, GBCPREDICT, GBCSAMPLE, GBCMETRICS.

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
    muX = mean(X,1);
    sdX = std(X,0,1);  sdX(sdX < eps) = 1;
    muY = mean(Y);
    sdY = std(Y);      if sdY < eps, sdY = 1; end
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

batch  = min(opts.MiniBatchSize, n);
nBatch = max(1, ceil(n/batch));   % keep the partial last batch
totalIter = opts.MaxEpochs * nBatch;

history = struct('epoch',[],'loss',[],'anchor',[],'ordering',[], ...
                 'pinball',[],'lr',[],'valCRPS',[]);
lossLog = zeros(opts.MaxEpochs,4);
lrLog   = zeros(opts.MaxEpochs,1);
valLog  = nan(opts.MaxEpochs,1);

hasVal = ~isempty(opts.ValidationData);
if hasVal
    Xval = opts.ValidationData{1};
    Yval = opts.ValidationData{2};
    tauGridVal = (1:99)/100;
end

if opts.Verbose
    fprintf('GBC-IQN: n=%d, d=%d, width=%d, epochs=%d, batch=%d, device=%s\n', ...
        n, d, opts.HiddenSize, opts.MaxEpochs, batch, ternary(useGPU,'gpu','cpu'));
    fprintf('%8s %12s %10s %10s %10s %10s\n', ...
        'epoch','loss','anchor','order','pinball','lr');
end

iter = 0;
t0 = tic;
for epoch = 1:opts.MaxEpochs

    idx = randperm(n);
    epochParts = zeros(1,3);
    epochLoss  = 0;

    for b = 1:nBatch
        iter = iter + 1;

        sel = idx((b-1)*batch + 1 : min(b*batch, n));
        Xb  = dlarray(Xall(:,sel), 'CB');
        Yb  = Yall(:,sel);

        % one tau draw per example (Algorithm 1, training phase)
        tau = rand(1, numel(sel), 'single');
        if useGPU, tau = gpuArray(tau); end
        Phi = dlarray(quantileEmbedding(tau, nh), 'CB');

        [lossVal, grads, parts] = dlfeval(@gbcLoss, params, Xb, Phi, tau, ...
                                          Yb, w, opts.L2Regularization);

        lr = cosineAnneal(opts.InitialLR, opts.MinLR, iter, totalIter);

        [params, avgG, avgSqG] = adamupdate(params, grads, avgG, avgSqG, ...
            iter, lr, opts.GradientDecay, opts.SqGradDecay);

        epochLoss  = epochLoss  + gather(double(extractdata(lossVal)));
        epochParts = epochParts + parts;
    end

    lossLog(epoch,:) = [epochLoss, epochParts] / nBatch;
    lrLog(epoch)     = lr;

    atCheckpoint = mod(epoch, opts.VerboseFreq) == 0 || epoch == 1 || ...
                   epoch == opts.MaxEpochs;

    if hasVal && atCheckpoint
        tmp = packModel(params, opts, muX, sdX, muY, sdY, d);
        valLog(epoch) = gbcCRPS(gbcPredict(tmp, Xval, tauGridVal), Yval);
    end

    if opts.Verbose && atCheckpoint
        if hasVal
            fprintf('%8d %12.5f %10.4f %10.4f %10.4f %10.2e   valCRPS %.4f\n', ...
                epoch, lossLog(epoch,1), lossLog(epoch,2), lossLog(epoch,3), ...
                lossLog(epoch,4), lr, valLog(epoch));
        else
            fprintf('%8d %12.5f %10.4f %10.4f %10.4f %10.2e\n', ...
                epoch, lossLog(epoch,1), lossLog(epoch,2), lossLog(epoch,3), ...
                lossLog(epoch,4), lr);
        end
    end
end

history.epoch    = (1:opts.MaxEpochs).';
history.loss     = lossLog(:,1);
history.anchor   = lossLog(:,2);
history.ordering = lossLog(:,3);
history.pinball  = lossLog(:,4);
history.lr       = lrLog;
history.valCRPS  = valLog;
history.trainTime = toc(t0);

model = packModel(params, opts, muX, sdX, muY, sdY, d);
model.history = history;
model.numTrain = n;

if opts.Verbose
    fprintf('Done in %.1f s.\n', history.trainTime);
end
end

% =========================================================================
function model = packModel(params, opts, muX, sdX, muY, sdY, d)
model = struct('params',params,'opts',opts,'muX',muX,'sdX',sdX, ...
               'muY',muY,'sdY',sdY,'dIn',d);
end

function lr = cosineAnneal(lrMax, lrMin, iter, total)
%COSINEANNEAL Loshchilov & Hutter (2017) single-cycle cosine schedule.
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
