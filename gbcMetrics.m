function [m, Q] = gbcMetrics(model, Xtest, Ytest, level, tauGrid)
%GBCMETRICS Held-out accuracy, calibration and sharpness of a GBC surrogate.
%
%   m = GBCMETRICS(model,Xtest,Ytest) reports the metrics used in the paper:
%   RMSE of the median prediction, CRPS, and empirical coverage / mean width
%   of the nominal 90% predictive interval.
%
%   m = GBCMETRICS(model,Xtest,Ytest,level) uses a different nominal interval
%   level (default 0.90).
%
%   m = GBCMETRICS(model,Xtest,Ytest,level,tauGrid) uses a custom quantile
%   grid for the CRPS estimator (default (1:99)/100).
%
%   Output fields
%     RMSE      root mean squared error of the predictive median
%     MAE       mean absolute error of the predictive median
%     CRPS      mean continuous ranked probability score
%     Coverage  fraction of Ytest inside the nominal interval
%     Width     mean width of that interval
%     Level     the nominal level used
%     PIT       n-by-1 probability integral transform values, for a rank
%               histogram; a calibrated model gives PIT ~ U[0,1]
%
%   See also GBCPREDICT, GBCCRPS.

arguments
    model   struct
    Xtest   (:,:) double
    Ytest   (:,1) double
    level   (1,1) double {mustBePositive} = 0.90
    tauGrid (1,:) double = (1:99)/100
end

if level <= 0 || level >= 1
    error('gbcMetrics:BadLevel','level must lie strictly in (0,1).');
end

lo = (1-level)/2;
hi = 1 - lo;

% Evaluate the CRPS grid together with the interval endpoints and the median
% in a single pass, then split the columns apart.
allTau = unique([tauGrid, lo, 0.5, hi]);
Qall   = gbcPredict(model, Xtest, allTau);

Q      = Qall(:, ismember(allTau, tauGrid));
med    = Qall(:, allTau == 0.5);
qLo    = Qall(:, allTau == lo);
qHi    = Qall(:, allTau == hi);

resid = Ytest - med;

m.RMSE     = sqrt(mean(resid.^2));
m.MAE      = mean(abs(resid));
m.CRPS     = gbcCRPS(Q, Ytest);
m.Coverage = mean(Ytest >= qLo & Ytest <= qHi);
m.Width    = mean(qHi - qLo);
m.Level    = level;
m.PIT      = mean(Q <= Ytest, 2);
end
