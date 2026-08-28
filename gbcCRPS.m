function [meanCRPS, perPoint] = gbcCRPS(Q, y)
%GBCCRPS Continuous ranked probability score from a quantile / sample matrix.
%
%   [meanCRPS, perPoint] = GBCCRPS(Q,y) evaluates the estimator used in
%   Polson & Sokolov (2026):
%
%       CRPS_hat(F,y) = (1/M) sum_m |q_m - y|
%                     - (1/(2 M^2)) sum_m sum_m' |q_m - q_m'|
%
%   Inputs
%     Q : n-by-M matrix; row i holds M quantiles (or samples) for test point i.
%     y : n-by-1 vector of observed responses.
%
%   Outputs
%     meanCRPS : mean score over the n test points (lower is better).
%     perPoint : n-by-1 vector of individual scores.
%
%   The double sum is evaluated in O(M log M) per row using the identity, for
%   q sorted ascending,
%
%       sum_i sum_j |q_i - q_j| = 2 * sum_i (2i - M - 1) q_(i),
%
%   so the second term equals (1/M^2) * sum_i (2i - M - 1) q_(i). This is
%   algebraically exact, not an approximation.
%
%   Note this is the standard biased (M-sample) estimator, matching the
%   paper; it is consistent as M grows and is the form used for the quantile
%   grid tau_m = m/(M+1).

arguments
    Q (:,:) double
    y (:,1) double
end

[n, M] = size(Q);
if n ~= numel(y)
    error('gbcCRPS:SizeMismatch','Q has %d rows but y has %d elements.', n, numel(y));
end
if M < 2
    error('gbcCRPS:TooFewQuantiles','Need at least two quantiles per point.');
end

Qs = sort(Q, 2);

term1 = mean(abs(Qs - y), 2);

i     = 1:M;
coef  = (2*i - M - 1);
term2 = (Qs * coef.') / M^2;

perPoint = term1 - term2;
meanCRPS = mean(perPoint);
end
