function m = gbcMetricsFromSamples(S, y, level, crpsMethod, tauSpan)
%GBCMETRICSFROMSAMPLES Accuracy, calibration and sharpness from a predictive
%   sample matrix.
%
%   m = GBCMETRICSFROMSAMPLES(S,y) treats each row of S as draws from the
%   predictive distribution of the corresponding element of y and reports the
%   metrics the paper uses.
%
%   m = GBCMETRICSFROMSAMPLES(S,y,level) sets the nominal interval level
%   (default 0.90).
%
%   m = GBCMETRICSFROMSAMPLES(S,y,level,"permuted") uses the reference's
%   single-permutation CRPS estimator instead of the exact pairwise one.
%
%   m = GBCMETRICSFROMSAMPLES(S,y,level,crpsMethod,tauSpan) corrects for the
%   grid the samples came from. IMPORTANT when S holds evaluated QUANTILES
%   rather than random draws: if the columns are the network's output at
%   levels spanning [a,b], then a raw empirical p-quantile of S lands at level
%   a + p*(b-a), not p. Pass tauSpan = [a b] and the interval endpoints are
%   taken at the preimage instead.
%
%   With the reference's grid, linspace(0.005,0.995,B), the uncorrected
%   endpoints sit at 0.0545 and 0.9455 - a 89.1% interval reported as 90%,
%   costing about 0.9 points of coverage. The distortion is affine, so it does
%   not shrink as B grows. Omit tauSpan to reproduce the reference's numbers
%   exactly; pass it to get the level you actually asked for.
%
%   Inputs
%     S       : n-by-B matrix of predictive draws or quantiles.
%     y       : n-by-1 observed responses.
%     tauSpan : [a b] span of the generating grid, or [] for none.
%
%   Output fields
%     RMSE, MAE, CRPS, Coverage, Width, Level, PIT
%
%   See also GBCMETRICS, GBCQUANTILE, GBCCRPS, GBCROWQUANTILE.

arguments
    S          (:,:) double
    y          (:,1) double
    level      (1,1) double = 0.90
    crpsMethod (1,1) string = "exact"
    tauSpan    (1,:) double = []
end

if size(S,1) ~= numel(y)
    error('gbcMetricsFromSamples:SizeMismatch', ...
          'S has %d rows but y has %d elements.', size(S,1), numel(y));
end
if level <= 0 || level >= 1
    error('gbcMetricsFromSamples:BadLevel','level must lie strictly in (0,1).');
end
if ~isempty(tauSpan) && numel(tauSpan) ~= 2
    error('gbcMetricsFromSamples:BadSpan','tauSpan must be [a b] or empty.');
end

lo = (1-level)/2;
hi = 1 - lo;
p  = [lo 0.5 hi];

if ~isempty(tauSpan)
    a = tauSpan(1); b = tauSpan(2);
    if b <= a
        error('gbcMetricsFromSamples:BadSpan','tauSpan must satisfy b > a.');
    end
    p = min(1, max(0, (p - a) ./ (b - a)));
end

Ss  = sort(S, 2);
qLo = gbcRowQuantile(Ss, p(1));
med = gbcRowQuantile(Ss, p(2));
qHi = gbcRowQuantile(Ss, p(3));

resid = y - med;

m.RMSE     = sqrt(mean(resid.^2));
m.MAE      = mean(abs(resid));
m.CRPS     = gbcCRPS(S, y, crpsMethod);
m.Coverage = mean(y >= qLo & y <= qHi);
m.Width    = mean(qHi - qLo);
m.Level    = level;
m.PIT      = mean(S <= y, 2);
end
