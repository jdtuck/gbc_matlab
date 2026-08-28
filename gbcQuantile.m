function Q = gbcQuantile(model, Xnew, probs, nGrid)
%GBCQUANTILE Predictive quantiles at the levels you asked for.
%
%   Q = GBCQUANTILE(model, Xnew, probs) returns an nNew-by-numel(probs) matrix
%   whose column j holds the probs(j) quantile of the predictive distribution
%   at each test input.
%
%   USE THIS, NOT GBCPREDICT, whenever you want specific levels - the 5% and
%   95% edges of an interval, a median line for a plot. GBCPREDICT returns one
%   column per grid point for a single model, but for an ENSEMBLE it pools
%   every member's columns, so asking it for [0.05 0.5 0.95] across K = 3
%   members gives 15 sorted columns, not 3, and column 5 is nowhere near the
%   95% level. GBCQUANTILE handles both cases correctly.
%
%   Q = GBCQUANTILE(model, Xnew, probs, nGrid) sets the internal grid
%   resolution used for ensembles (default 512).
%
%   Inputs
%     model : struct from GBCTRAIN, or cell array from GBCENSEMBLE.
%     Xnew  : nNew-by-d test inputs.
%     probs : vector of levels in (0,1).
%     nGrid : ensemble grid resolution.
%
%   For a single model the network is queried at the requested levels
%   directly, so the answer is exact. For an ensemble the predictive law is a
%   mixture over members and has no closed form, so each member is evaluated
%   on a dense midpoint grid tau_g = (g-0.5)/G, the columns are pooled, and
%   the empirical quantile of that pooled sample is taken - with the grid's
%   affine level distortion undone (see GBCROWQUANTILE), which is worth about
%   a full point of coverage at the 90% level.
%
%   See also GBCPREDICT, GBCENSEMBLE, GBCMETRICSFROMSAMPLES.

arguments
    model {mustBeA(model,["struct","cell","gbcModel"])}
    Xnew  (:,:) double
    probs (1,:) double
    nGrid (1,1) double {mustBePositive, mustBeInteger} = 512
end

if any(probs <= 0 | probs >= 1)
    error('gbcQuantile:BadProbs','Levels must lie strictly in (0,1).');
end

% ---- single model: ask the network for exactly these levels --------------
if ~iscell(model)
    Q = gbcPredict(model, Xnew, probs);
    return
end

% ---- ensemble: quantile of the pooled mixture ---------------------------
G    = nGrid;
grid = ((1:G) - 0.5) / G;                 % midpoint rule, spans [1/2G, 1-1/2G]

S  = gbcPredict(model, Xnew, grid);       % nNew-by-(K*G)
Ss = sort(S, 2);

% Undo the affine distortion: an empirical p-quantile of a sample drawn from
% a grid spanning [a,b] sits at level a + p*(b-a), so query at the preimage.
a = grid(1);
b = grid(end);
pp = min(1, max(0, (probs - a) ./ (b - a)));

Q = zeros(size(Xnew,1), numel(probs));
for j = 1:numel(probs)
    Q(:,j) = gbcRowQuantile(Ss, pp(j));
end
end
