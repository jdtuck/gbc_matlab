function [m, Q] = gbcMetrics(model, Xtest, Ytest, level, tauGrid, crpsMethod)
%GBCMETRICS Held-out accuracy, calibration and sharpness of a GBC surrogate.
%
%   m = GBCMETRICS(model,Xtest,Ytest) evaluates the model on a 99-point
%   quantile grid and reports RMSE of the predictive median, CRPS, and the
%   empirical coverage and mean width of the nominal 90% predictive interval.
%
%   m = GBCMETRICS(model,Xtest,Ytest,level) sets the interval level.
%   m = GBCMETRICS(model,Xtest,Ytest,level,tauGrid) sets the quantile grid.
%   m = GBCMETRICS(...,crpsMethod) selects "exact" (default) or "permuted";
%   see GBCCRPS.
%
%   model may be a single model or a cell array of models (see GBCENSEMBLE).
%
%   This is a convenience wrapper: it builds the predictive sample matrix with
%   GBCPREDICT and hands it to GBCMETRICSFROMSAMPLES. Call that directly if
%   you already have samples.
%
%   See also GBCMETRICSFROMSAMPLES, GBCPREDICT, GBCCRPS.

arguments
    model      {mustBeA(model,["struct","cell","gbcModel"])}
    Xtest      (:,:) double
    Ytest      (:,1) double
    level      (1,1) double = 0.90
    tauGrid    (1,:) double = (1:99)/100
    crpsMethod (1,1) string = "exact"
end

Q = gbcPredict(model, Xtest, tauGrid);
m = gbcMetricsFromSamples(Q, Ytest, level, crpsMethod);
end
