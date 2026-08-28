%% GBC on the 10-dimensional Friedman function
% Reproduces the accuracy setting of Polson & Sokolov (2026), Sec. 5:
%
%   f(x) = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5,   x in [0,1]^10
%   y    = f(x) + eps,  eps ~ N(0,1)
%
% Five active inputs, five inert ones. Training set n = 2000, test set 500.
% The paper reports GBC improving CRPS by roughly 14% over a GP surrogate
% here; this script reports GBC's own numbers plus, for reference, an
% oracle CRPS computed from the true N(f(x),1) predictive distribution -
% the best any method could achieve on this data-generating process.
%
% Runtime: a couple of minutes on CPU at the default 1500 epochs.

clear; close all;
addpath(fileparts(fileparts(mfilename('fullpath'))));

rng(1);

%% ---------------------------------------------------------------- data
d      = 10;
nTrain = 2000;
nTest  = 500;
sigma  = 1;

friedman = @(X) 10*sin(pi*X(:,1).*X(:,2)) + 20*(X(:,3)-0.5).^2 + ...
                10*X(:,4) + 5*X(:,5);

Xtr = rand(nTrain, d);
Ytr = friedman(Xtr) + sigma*randn(nTrain,1);

Xte = rand(nTest, d);
fTe = friedman(Xte);
Yte = fTe + sigma*randn(nTest,1);

%% ---------------------------------------------------------------- train
opts = gbcOptions( ...
    'MaxEpochs',     1500, ...      % paper uses 3000-8000; 1500 suffices here
    'MiniBatchSize', 256, ...
    'LossWeights',   [0.3 0.3 0.4], ...   % smooth-response defaults
    'VerboseFreq',   100, ...
    'ValidationData',{Xte, Yte}, ...
    'Seed',          1);

model = gbcTrain(Xtr, Ytr, opts);

%% ---------------------------------------------------------------- evaluate
m = gbcMetrics(model, Xte, Yte);

% Oracle: the true predictive law is N(f(x), sigma^2), so its CRPS has a
% closed form. This is the floor, not a competing method.
zStd      = (Yte - fTe)/sigma;
oracle    = sigma*( zStd.*(2*normcdfLocal(zStd)-1) + 2*normpdfLocal(zStd) - 1/sqrt(pi) );
oracleCRPS = mean(oracle);

fprintf('\n--- Friedman 10-D, n_train = %d ---\n', nTrain);
fprintf('RMSE (median)      : %.4f\n', m.RMSE);
fprintf('CRPS               : %.4f   (oracle floor %.4f)\n', m.CRPS, oracleCRPS);
fprintf('90%% coverage       : %.3f   (nominal 0.900)\n', m.Coverage);
fprintf('90%% mean width     : %.4f   (oracle %.4f)\n', m.Width, 2*1.6449*sigma);

%% ---------------------------------------------------------------- plots
Q = gbcPredict(model, Xte, [0.05 0.5 0.95]);

figure('Name','GBC - Friedman 10D','Position',[100 100 1100 760]);

subplot(2,2,1);
errLo = Q(:,2)-Q(:,1); errHi = Q(:,3)-Q(:,2);
errorbar(fTe, Q(:,2), errLo, errHi, 'o', 'MarkerSize',3, 'CapSize',0, ...
    'Color',[0.3 0.5 0.8 0.35], 'MarkerFaceColor',[0.2 0.35 0.6]);
hold on; lims = [min(fTe) max(fTe)];
plot(lims, lims, 'k--', 'LineWidth',1.2);
xlabel('true f(x)'); ylabel('predicted median with 90% interval');
title('Calibration of the predictive interval'); axis tight; grid on;

subplot(2,2,2);
histogram(m.PIT, 20, 'Normalization','pdf', 'FaceColor',[0.35 0.55 0.75]);
hold on; yline(1,'k--','LineWidth',1.2);
xlabel('PIT'); ylabel('density');
title('Rank histogram (flat = calibrated)');

subplot(2,2,3);
semilogy(model.history.epoch, model.history.loss, 'LineWidth',1.2); hold on;
semilogy(model.history.epoch, model.history.pinball, 'LineWidth',1);
legend({'total','pinball term'},'Location','northeast');
xlabel('epoch'); ylabel('loss'); title('Training loss'); grid on;

% Predictive band along x1 with the other inputs held at their midpoint.
subplot(2,2,4);
g   = linspace(0,1,200).';
Xg  = [g, repmat(0.5, numel(g), d-1)];
Qg  = gbcPredict(model, Xg, [0.05 0.25 0.5 0.75 0.95]);
fill([g; flipud(g)], [Qg(:,1); flipud(Qg(:,5))], [0.75 0.83 0.92], ...
     'EdgeColor','none'); hold on;
fill([g; flipud(g)], [Qg(:,2); flipud(Qg(:,4))], [0.55 0.68 0.85], ...
     'EdgeColor','none');
plot(g, Qg(:,3), 'Color',[0.15 0.25 0.45], 'LineWidth',1.8);
plot(g, friedman(Xg), 'r--', 'LineWidth',1.5);
legend({'90% band','50% band','GBC median','truth'},'Location','southeast');
xlabel('x_1  (x_2..x_{10} = 0.5)'); ylabel('y'); title('Predictive slice'); grid on;

%% ---------------------------------------------------------------- helpers
function p = normcdfLocal(z)
p = 0.5*erfc(-z/sqrt(2));
end
function p = normpdfLocal(z)
p = exp(-0.5*z.^2)/sqrt(2*pi);
end
