function models = gbcEnsemble(X, Y, K, opts, seeds)
%GBCENSEMBLE Train K independent IQNs whose quantiles are pooled at test time.
%
%   models = GBCENSEMBLE(X,Y,K,opts) trains K networks that differ only in
%   their random seed, and returns them as a 1-by-K cell array. Pass that cell
%   array straight to GBCPREDICT or GBCSAMPLE: the columns of every member are
%   concatenated, so K members evaluated on an M-point quantile grid give a
%   pooled predictive sample of K*M values per test point.
%
%   models = GBCENSEMBLE(X,Y,K,opts,seeds) uses the supplied seed vector
%   instead of 1:K. The paper's motorcycle table uses seeds rep*13 + k*1000
%   for replicate rep and member k.
%
%   Ensembling is how the paper reaches its Table 1 numbers (K = 5, 100
%   quantiles each, 500 pooled samples per test point). It matters more than
%   it might seem on a dataset this small: a single IQN fit to 106 training
%   points has visible seed-to-seed variation, and pooling averages over the
%   quantile curve rather than over point predictions, so the ensemble is a
%   genuine mixture distribution rather than a smoothed mean.
%
%   See also GBCTRAIN, GBCPREDICT, GBCMETRICSFROMSAMPLES.

arguments
    X     (:,:) double
    Y     (:,1) double
    K     (1,1) double {mustBePositive, mustBeInteger}
    opts  struct = gbcOptions()
    seeds (1,:) double = []
end

if isempty(seeds)
    seeds = 1:K;
end
if numel(seeds) ~= K
    error('gbcEnsemble:BadSeeds','Expected %d seeds, got %d.', K, numel(seeds));
end

models = cell(1,K);
for k = 1:K
    o = opts;
    o.Seed = seeds(k);
    if opts.Verbose
        fprintf('-- ensemble member %d/%d (seed %d)\n', k, K, seeds(k));
    end
    models{k} = gbcTrain(X, Y, o);
end
end
