function m = gbcMetricsFromSamples(S, y, level, crpsMethod)
%GBCMETRICSFROMSAMPLES Accuracy, calibration and sharpness from a predictive
%   sample matrix.
%
%   m = GBCMETRICSFROMSAMPLES(S,y) treats each row of S as a set of draws from
%   the predictive distribution of the corresponding element of y, and reports
%   the metrics the paper uses. This is the form the reference implementation
%   evaluates: it takes empirical quantiles of the pooled samples rather than
%   querying the network at specific tau levels, which is what an ensemble
%   requires anyway.
%
%   m = GBCMETRICSFROMSAMPLES(S,y,level) sets the nominal interval level
%   (default 0.90).
%
%   m = GBCMETRICSFROMSAMPLES(S,y,level,"permuted") uses the reference's
%   single-permutation CRPS estimator instead of the exact pairwise one. See
%   GBCCRPS for what differs.
%
%   Inputs
%     S : n-by-B matrix of predictive draws (or quantiles).
%     y : n-by-1 vector of observed responses.
%
%   Output fields
%     RMSE      root mean squared error of the predictive median
%     MAE       mean absolute error of the predictive median
%     CRPS      mean continuous ranked probability score
%     Coverage  fraction of y inside the nominal interval
%     Width     mean width of that interval
%     Level     the nominal level used
%     PIT       n-by-1 probability integral transform values
%
%   See also GBCMETRICS, GBCCRPS, GBCENSEMBLE.

arguments
    S          (:,:) double
    y          (:,1) double
    level      (1,1) double = 0.90
    crpsMethod (1,1) string = "exact"
end

if size(S,1) ~= numel(y)
    error('gbcMetricsFromSamples:SizeMismatch', ...
          'S has %d rows but y has %d elements.', size(S,1), numel(y));
end
if level <= 0 || level >= 1
    error('gbcMetricsFromSamples:BadLevel','level must lie strictly in (0,1).');
end

lo = (1-level)/2;
hi = 1 - lo;

Ss  = sort(S, 2);
med = rowQuantile(Ss, 0.5);
qLo = rowQuantile(Ss, lo);
qHi = rowQuantile(Ss, hi);

resid = y - med;

m.RMSE     = sqrt(mean(resid.^2));
m.MAE      = mean(abs(resid));
m.CRPS     = gbcCRPS(S, y, crpsMethod);
m.Coverage = mean(y >= qLo & y <= qHi);
m.Width    = mean(qHi - qLo);
m.Level    = level;
m.PIT      = mean(S <= y, 2);
end

% -------------------------------------------------------------------------
function v = rowQuantile(Ss, p)
%ROWQUANTILE Linear-interpolation quantile of each row of an already-sorted
%   matrix. Matches numpy.quantile's default ('linear') method, which is what
%   the reference uses, and avoids a Statistics Toolbox dependency.
B = size(Ss,2);
pos = p*(B-1) + 1;                 % 1-based fractional position
loI = floor(pos);
hiI = min(loI+1, B);
frac = pos - loI;
v = (1-frac)*Ss(:,loI) + frac*Ss(:,hiI);
end
