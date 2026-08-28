function S = gbcSample(model, Xnew, B)
%GBCSAMPLE Draw predictive samples from the GBC surrogate.
%
%   S = GBCSAMPLE(model,Xnew,B) implements the test phase of Algorithm 1:
%
%       for b = 1..B:  tau^(b) ~ U[0,1],  y^(b) = q_hat_{tau^(b)}(x*)
%
%   Each test point receives its own independent tau draws, so the columns of
%   S are exchangeable draws from the estimated conditional distribution
%   rather than a shared quantile path.
%
%   Inputs
%     model : struct from GBCTRAIN.
%     Xnew  : nNew-by-d matrix of test inputs.
%     B     : number of predictive draws per input (default 200).
%
%   Output
%     S : nNew-by-B matrix of predictive samples on the original scale.
%
%   Because a single forward pass yields a draw, generating a full predictive
%   distribution costs B network evaluations and no linear algebra - this is
%   the O(n) test-time behaviour that motivates GBC over a GP surrogate.
%
%   See also GBCPREDICT, GBCTRAIN.

%   ENSEMBLES. If model is a cell array (see GBCENSEMBLE), each of the K
%   members draws B samples and the results are pooled, giving n-by-(K*B).

arguments
    model {mustBeA(model,["struct","cell","gbcModel"])}
    Xnew  (:,:) double
    B     (1,1) double {mustBePositive, mustBeInteger} = 200
end

if iscell(model)
    parts = cell(1,numel(model));
    for k = 1:numel(model)
        parts{k} = gbcSample(model{k}, Xnew, B);
    end
    S = [parts{:}];
    return
end

if size(Xnew,2) ~= model.dIn
    error('gbcSample:BadWidth','Model expects %d predictors, got %d.', ...
          model.dIn, size(Xnew,2));
end

nh    = model.opts.NumCosine;
n     = size(Xnew,1);
onGPU = false;
try, onGPU = isgpuarray(extractdata(model.params.Wx)); catch, end

Xs = (Xnew - model.muX) ./ model.sdX;
X0 = single(Xs.');                        % d-by-n
if onGPU, X0 = gpuArray(X0); end

% Draws are batched the same way GBCPREDICT batches quantile levels: tile the
% test points across draws so one forward pass yields many samples.
maxCols = max(1000, round(5e6 / max(1, model.opts.HiddenSize)));
T       = max(1, min(B, floor(maxCols / max(1,n))));

S = zeros(n, B);
for s = 1:T:B
    blk = s:min(s+T-1, B);
    nb  = numel(blk);

    Xrep = repmat(X0, 1, nb);
    tRep = rand(1, n*nb, 'single');        % independent tau per (point, draw)
    if onGPU, tRep = gpuArray(tRep); end

    Xd  = dlarray(Xrep, 'CB');
    Phi = dlarray(quantileEmbedding(tRep, nh), 'CB');

    [~, qHat] = gbcForward(model.params, Xd, Phi);

    S(:,blk) = reshape(gather(double(extractdata(qHat))), n, nb);
end

S = S * model.sdY + model.muY;
end
