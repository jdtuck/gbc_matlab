function Q = gbcPredict(model, Xnew, tauGrid, rearrange)
%GBCPREDICT Conditional quantiles of the GBC surrogate.
%
%   Q = GBCPREDICT(model,Xnew) returns predictive quantiles at the grid
%   tau_m = m/100, m = 1..99.
%
%   Q = GBCPREDICT(model,Xnew,tauGrid) evaluates the requested levels. The
%   reference implementation samples on linspace(0.005,0.995,B) with B = 500;
%   pass that grid to match it.
%
%   Q = GBCPREDICT(model,Xnew,tauGrid,false) disables quantile rearrangement.
%
%   ENSEMBLES. If model is a cell array of models (see GBCENSEMBLE), every
%   member is evaluated on the grid and the columns are pooled, giving an
%   n-by-(K*M) matrix of predictive draws sorted ascending along each row.
%   Column order is then the sorted pooled sample, not the caller's tau order.
%
%   Inputs
%     model    : struct from GBCTRAIN, or 1-by-K cell array from GBCENSEMBLE.
%     Xnew     : nNew-by-d matrix of test inputs.
%     tauGrid  : vector of quantile levels in (0,1). Default (1:99)/100.
%     rearrange: enforce monotonicity in tau. Default true. This is the
%                Chernozhukov-Fernandez-Galichon rearrangement; it is an
%                addition to the paper and the reference, both of which rely
%                on the ordering surrogate in the loss to discourage (but not
%                forbid) crossings. Rearranging never increases the aggregate
%                check loss of the quantile curve (verified in
%                tests/test_gbc.m), and it leaves the CRPS estimator
%                unchanged, since that estimator is permutation invariant.
%
%   Output
%     Q : nNew-by-numel(tauGrid) matrix of quantiles on the original scale,
%         with columns in the same order as the tauGrid you supplied (single
%         model), or the pooled sorted sample (ensemble).
%
%   See also GBCTRAIN, GBCENSEMBLE, GBCSAMPLE, GBCMETRICS, GBCCRPS.

arguments
    model     {mustBeA(model,["struct","cell"])}
    Xnew      (:,:) double
    tauGrid   (1,:) double = (1:99)/100
    rearrange (1,1) logical = true
end

% ------------------------------------------------------------ ensembles
if iscell(model)
    K = numel(model);
    parts = cell(1,K);
    for k = 1:K
        parts{k} = gbcPredict(model{k}, Xnew, tauGrid, rearrange);
    end
    Q = sort([parts{:}], 2);
    return
end

if size(Xnew,2) ~= model.dIn
    error('gbcPredict:BadWidth','Model expects %d predictors, got %d.', ...
          model.dIn, size(Xnew,2));
end
if any(tauGrid <= 0 | tauGrid >= 1)
    error('gbcPredict:BadTau','Quantile levels must lie strictly in (0,1).');
end

% Work on an ascending grid so that rearrangement is well defined, then
% restore the caller's column order.
[tauSorted, order] = sort(tauGrid);

nh = model.opts.NumCosine;
n  = size(Xnew,1);
M  = numel(tauSorted);
onGPU = paramsOnGPU(model.params);

Xs = (Xnew - model.muX) ./ model.sdX;
Xd = single(Xs.');
if onGPU, Xd = gpuArray(Xd); end
Xd = dlarray(Xd,'CB');

Qs = zeros(n, M);
for m = 1:M
    tau = repmat(single(tauSorted(m)), 1, n);
    if onGPU, tau = gpuArray(tau); end
    Phi = dlarray(quantileEmbedding(tau, nh), 'CB');

    [~, qHat] = gbcForward(model.params, Xd, Phi);
    Qs(:,m) = gather(double(extractdata(qHat))).';
end

Qs = Qs * model.sdY + model.muY;

if rearrange
    Qs = sort(Qs, 2);
end

Q = zeros(n, M);
Q(:, order) = Qs;
end

% -------------------------------------------------------------------------
function tf = paramsOnGPU(params)
tf = false;
try
    tf = isgpuarray(extractdata(params.Wx));
catch
    try
        tf = isa(extractdata(params.Wx),'gpuArray');
    catch
        tf = false;
    end
end
end
