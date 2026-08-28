function [meanCRPS, perPoint] = gbcCRPS(Q, y, method)
%GBCCRPS Continuous ranked probability score from a quantile / sample matrix.
%
%   [meanCRPS, perPoint] = GBCCRPS(Q,y) evaluates the estimator written in the
%   paper,
%
%       CRPS_hat(F,y) = (1/M) sum_m |q_m - y|
%                     - (1/(2 M^2)) sum_m sum_m' |q_m - q_m'|
%
%   GBCCRPS(Q,y,"permuted") instead evaluates the estimator the authors'
%   reference code uses, which replaces the full double sum by a single random
%   pairing:
%
%       CRPS_hat(F,y) = mean_m |q_m - y| - 0.5 * mean_m |q_m - q_{perm(m)}|
%
%   Both estimate the same quantity, E|Y-y| - 0.5*E|Y-Y'|. The permuted form
%   is O(M) rather than O(M log M) and is what the published numbers were
%   computed with, but it is stochastic: repeated calls on identical inputs
%   give slightly different answers, and its variance is materially higher.
%   "exact" is the default because it is deterministic and strictly lower
%   variance; use "permuted" only when matching the reference number for
%   number.
%
%   Inputs
%     Q      : n-by-M matrix; row i holds M quantiles (or samples) for point i.
%     y      : n-by-1 vector of observed responses.
%     method : "exact" (default) or "permuted".
%
%   Outputs
%     meanCRPS : mean score over the n test points (lower is better).
%     perPoint : n-by-1 vector of individual scores.
%
%   The exact double sum is evaluated in O(M log M) per row using the
%   identity, for q sorted ascending,
%
%       sum_i sum_j |q_i - q_j| = 2 * sum_i (2i - M - 1) q_(i),
%
%   so the second term equals (1/M^2) * sum_i (2i - M - 1) q_(i). This is
%   algebraically exact, not an approximation.
%
%   Note the exact form is the standard biased (M-sample) estimator, matching
%   the paper; it is consistent as M grows.

arguments
    Q      (:,:) double
    y      (:,1) double
    method (1,1) string = "exact"
end

[n, M] = size(Q);
if n ~= numel(y)
    error('gbcCRPS:SizeMismatch','Q has %d rows but y has %d elements.', n, numel(y));
end
if M < 2
    error('gbcCRPS:TooFewQuantiles','Need at least two quantiles per point.');
end

switch lower(method)
    case "exact"
        Qs    = sort(Q, 2);
        term1 = mean(abs(Qs - y), 2);
        i     = 1:M;
        coef  = (2*i - M - 1);
        term2 = (Qs * coef.') / M^2;

    case "permuted"
        % Reference: term2 = 0.5 * mean_m |q_m - q_{perm(m)}|, one shared
        % permutation of the sample index across all test points.
        perm  = randperm(M);
        term1 = mean(abs(Q - y), 2);
        term2 = 0.5 * mean(abs(Q - Q(:,perm)), 2);

    otherwise
        error('gbcCRPS:BadMethod','method must be "exact" or "permuted".');
end

perPoint = term1 - term2;
meanCRPS = mean(perPoint);
end
