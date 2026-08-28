%% GBC on the motorcycle crash data (MASS::mcycle) - the paper's Table 1
%
% The canonical heteroskedastic benchmark: head acceleration measured after a
% simulated motorcycle crash, n = 133, d = 1. Three regimes in one series -
% an almost noiseless pre-impact phase, a violent and highly variable
% deceleration, and a moderately noisy rebound. The conditional variance
% changes by more than an order of magnitude across the range, which is why
% the paper benchmarks against hetGP here rather than an ordinary GP: a
% homoskedastic surrogate has to pick one noise level and be wrong nearly
% everywhere.
%
% GBC needs no variance model at all. It learns the whole conditional
% quantile function, so changing spread is just a change in the spacing of
% the quantile curves - no separate noise process, no kernel for it.
%
%% Two presets
%
% PRESET = "paper" reproduces experiments/tab1_motorcycle.py exactly:
%   50 splits x 5 ensemble members x 5000 full-batch steps = 1.25M steps.
%   That is genuinely expensive - the authors quote ~30 min for this table,
%   and MATLAB's per-call dlfeval overhead makes it slower still. Use it when
%   you want the table, not when you want to see the method work.
%
% PRESET = "fast" (default) costs a small fraction of that and gives a
%   visually identical fit. The saving is not just "fewer steps": the
%   reference draws ONE tau per gradient step and shares it across the batch,
%   so 5000 steps only ever visit 5000 quantile levels. Setting
%   TauPerExample = true draws an independent tau for every training point,
%   so each step visits ~106 levels instead of 1. Quantile coverage per step
%   goes up by about two orders of magnitude, and far fewer steps are needed
%   to resolve the same curve.
%
%   This is a deliberate departure from the reference protocol. The model,
%   the loss and the data are identical; only the tau sampling and the step
%   count change. Numbers will differ slightly from the paper's. Use "paper"
%   if you need to match it.
%
% NOTE ON EXACT REPRODUCTION: even under "paper", replicate splits come from
% MATLAB's RNG rather than numpy's, so individual replicate numbers will not
% match the paper. The aggregate should be comparable.

clear; close all;
addpath(fileparts(fileparts(mfilename('fullpath'))));

PRESET = "fast";       % "fast" | "paper"

switch PRESET
    case "fast"
        NREP = 5; K = 3; EPOCHS = 1200; BPER = 100; PEREX = true;
    case "paper"
        NREP = 50; K = 5; EPOCHS = 5000; BPER = 100; PEREX = false;
    otherwise
        error('demo_motorcycle:BadPreset','PRESET must be "fast" or "paper".');
end

%% ---------------------------------------------------------------- data
[t, y] = mcycleData();
n = numel(t);

fprintf('mcycle: n = %d, times %.1f-%.1f ms, accel %.1f to %.1f g\n', ...
        n, min(t), max(t), min(y), max(y));
fprintf('preset "%s": %d replicates x %d members x %d steps = %s gradient steps\n', ...
        PRESET, NREP, K, EPOCHS, addCommas(NREP*K*EPOCHS));
fprintf('tau sampling: %s\n\n', ...
        ternary(PEREX,'independent per training point','one per step (reference)'));

%% ---------------------------------------------------------------- replicates
tauGrid = linspace(0.005, 0.995, BPER);   % the reference's sampling grid

rmseAll = zeros(NREP,1);
crpsAll = zeros(NREP,1);
covAll  = zeros(NREP,1);
widAll  = zeros(NREP,1);
pitAll  = [];

opts = gbcOptions( ...
    'MaxEpochs',     EPOCHS, ...
    'TauPerExample', PEREX, ...
    'LossWeights',   [0.3 0.3 0.4], ...   % smooth/heteroskedastic defaults
    'Verbose',       false);

fprintf('%5s %10s %10s %10s %10s %12s\n', ...
        'rep','RMSE','CRPS','cover90','width90','elapsed');
tStart = tic;
for rep = 1:NREP

    rng(rep + 300);                       % split seed, per the reference
    idx  = randperm(n);
    nTr  = round(0.8*n);
    tr   = idx(1:nTr);
    te   = idx(nTr+1:end);

    seeds  = rep*13 + (1:K)*1000;         % member seeds, per the reference
    models = gbcEnsemble(t(tr), y(tr), K, opts, seeds);

    S = gbcPredict(models, t(te), tauGrid);   % n_test-by-(K*BPER) pooled
    % Pass the grid span: without it the "90%" endpoints would silently be
    % taken at tau 0.0545 / 0.9455, an 89.1% interval.
    m = gbcMetricsFromSamples(S, y(te), 0.90, "exact", ...
                              [tauGrid(1) tauGrid(end)]);

    rmseAll(rep) = m.RMSE;
    crpsAll(rep) = m.CRPS;
    covAll(rep)  = m.Coverage;
    widAll(rep)  = m.Width;
    pitAll       = [pitAll; m.PIT]; %#ok<AGROW>

    el = toc(tStart);
    fprintf('%5d %10.3f %10.3f %10.3f %10.2f %9.1f s\n', ...
            rep, m.RMSE, m.CRPS, m.Coverage, m.Width, el);

    % Project the total after the first replicate, so a long run announces
    % its cost early instead of going quiet.
    if rep == 1 && NREP > 1
        fprintf('      (~%.1f s per replicate; ~%.1f min for all %d)\n', ...
                el, el*NREP/60, NREP);
    end
end
elapsed = toc(tStart);

fprintf('\n--- Motorcycle, %d replicates, K = %d, %d pooled samples ---\n', ...
        NREP, K, K*BPER);
report('RMSE',        rmseAll);
report('CRPS',        crpsAll);
report('90%% coverage',covAll);
report('90%% width',   widAll);
fprintf('(%.1f s total, %.1f s per replicate)\n', elapsed, elapsed/NREP);

%% ---------------------------------------------------------------- full fit
% One ensemble on all 133 points, purely for the picture.
fprintf('\nFitting a display ensemble on all %d points...\n', n);
final = gbcEnsemble(t, y, K, opts, 900 + (1:K)*7);

tg = linspace(min(t), max(t), 400).';

% gbcQuantile, NOT gbcPredict. For an ensemble, gbcPredict pools every
% member's columns, so a 5-level request across K members returns K*5 sorted
% columns and Q(:,5) is not the 95% level - it lands around the 25th
% percentile, which draws a "90% band" that misses most of the data.
% gbcQuantile returns one column per requested level for both cases.
Q = gbcQuantile(final, tg, [0.05 0.25 0.5 0.75 0.95]);

% Sanity check the band we are about to draw, rather than trusting the plot.
Qobs = gbcQuantile(final, t, [0.05 0.95]);
bandCover = mean(y >= Qobs(:,1) & y <= Qobs(:,2));
fprintf('plotted 90%% band covers %.1f%% of the %d observations (in-sample)\n', ...
        100*bandCover, n);

figure('Name','GBC - motorcycle crash','Position',[80 80 1150 780]);

% --- the classic fan chart
subplot(2,2,[1 2]);
fill([tg; flipud(tg)], [Q(:,1); flipud(Q(:,5))], [0.78 0.85 0.93], ...
     'EdgeColor','none'); hold on;
fill([tg; flipud(tg)], [Q(:,2); flipud(Q(:,4))], [0.55 0.68 0.86], ...
     'EdgeColor','none');
plot(tg, Q(:,3), 'Color',[0.12 0.22 0.42], 'LineWidth',2);
scatter(t, y, 26, [0.15 0.15 0.15], 'filled', 'MarkerFaceAlpha',0.65);
xlabel('time after impact (ms)'); ylabel('head acceleration (g)');
title('GBC predictive quantiles - the band narrows and widens with the data');
legend({'90% band','50% band','median','observations'}, 'Location','southeast');
grid on; axis tight;

% --- the point of the benchmark: the interval width is not constant
subplot(2,2,3);
plot(tg, Q(:,5)-Q(:,1), 'Color',[0.15 0.35 0.6], 'LineWidth',1.8);
xlabel('time after impact (ms)'); ylabel('90% interval width (g)');
title('Learned heteroskedasticity (no variance model)');
grid on; axis tight;

% --- calibration: pooled PIT over every held-out point of every replicate.
% A calibrated predictive distribution gives PIT ~ Uniform[0,1], so this
% histogram should be flat. A hump in the middle means intervals that are too
% wide; peaks at the ends mean too narrow.
subplot(2,2,4);
histogram(pitAll, 12, 'Normalization','pdf', 'FaceColor',[0.45 0.62 0.82], ...
    'EdgeColor','w');
hold on; yline(1, 'k--', 'LineWidth',1.4);
xlabel('PIT'); ylabel('density'); xlim([0 1]);
title(sprintf('Rank histogram, %d held-out points', numel(pitAll)));
grid on;

%% ---------------------------------------------------------------- helpers
function report(name, v)
fprintf(['  ' name ': %.4f +/- %.4f  (mean +/- SE over %d reps)\n'], ...
        mean(v), stderr(v), numel(v));
end

function s = stderr(v)
% Standard error with the population SD, matching the reference's
% np.nanstd(a)/sqrt(n).
s = std(v,1) / sqrt(numel(v));
end

function s = addCommas(x)
s = regexprep(sprintf('%d',round(x)), '(\d)(?=(\d{3})+$)', '$1,');
end

function out = ternary(c,a,b)
if c, out = a; else, out = b; end
end
