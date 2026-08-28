%% GBC on a piecewise jump-process (BGP) benchmark, d = 2
% Reproduces the "bi-mixture GP" family of Polson & Sokolov (2026), Sec. 5:
%
%   f(x) = f1(x) 1{a'x >= 0} + f2(x) 1{a'x < 0},     x in [-0.5, 0.5]^d
%   f1 ~ GP(0,  9 k_SE),   f2 ~ GP(13, 9 k_SE),      length scale 0.1 d
%   y  = f(x) + eps,       eps ~ N(0, 4)
%
% The regime orientation a is drawn from {-1,+1}^d. The jump-to-noise ratio
% is 13/2 = 6.5, matching the paper. This is the setting where GBC's
% distributional output pays off: near the boundary the conditional law is
% genuinely bimodal, which a Gaussian-predictive GP cannot represent no
% matter how the kernel is tuned.
%
% Note the loss weights: the paper's "quantile-dominant" setting
% (0.1, 0.2, 0.7) is used, which it reports improves CRPS by ~28% on jump
% processes relative to the smooth-response defaults. Set USE_SMOOTH_WEIGHTS
% below to true to see that comparison yourself.

clear; close all;
addpath(fileparts(fileparts(mfilename('fullpath'))));

rng(7);

USE_SMOOTH_WEIGHTS = false;

%% ---------------------------------------------------------------- data
d       = 2;
N       = 2000;            % total; 80:20 split as in the paper
ell     = 0.1*d;           % SE length scale
sigmaF2 = 9;               % GP variance
jump    = 13;              % mean of the second component
noiseSD = 2;               % eps ~ N(0,4)

X = rand(N,d) - 0.5;
a = sign(randn(1,d)); a(a==0) = 1;

% Joint draw of the two GP components over all N design points.
K = sigmaF2 * exp(-pdist2sq(X,X)/(2*ell^2));

% SE kernels are badly conditioned at this length scale; grow the jitter until
% the factorisation succeeds rather than failing outright.
jit = 1e-8*sigmaF2;
while true
    [L, p] = chol(K + jit*eye(N), 'lower');
    if p == 0, break; end
    jit = jit*10;
    if jit > sigmaF2
        error('demo_jump2d:Chol','Could not factorise the kernel matrix.');
    end
end

f1 = L*randn(N,1);
f2 = jump + L*randn(N,1);

inRegime1 = (X*a.') >= 0;
f = f1.*inRegime1 + f2.*(~inRegime1);
Y = f + noiseSD*randn(N,1);

idx  = randperm(N);
nTr  = round(0.8*N);
tr   = idx(1:nTr);  te = idx(nTr+1:end);
Xtr  = X(tr,:);  Ytr = Y(tr);
Xte  = X(te,:);  Yte = Y(te);

%% ---------------------------------------------------------------- train
if USE_SMOOTH_WEIGHTS
    w = [0.3 0.3 0.4];   tag = 'smooth defaults';
else
    w = [0.1 0.2 0.7];   tag = 'quantile-dominant';
end

opts = gbcOptions( ...
    'MaxEpochs',      3000, ...           % the reference's default
    'LossWeights',    w, ...
    'VerboseFreq',    250, ...
    'ValidationData', {Xte, Yte}, ...
    'Seed',           7);

model = gbcTrain(Xtr, Ytr, opts);

%% ---------------------------------------------------------------- evaluate
m = gbcMetrics(model, Xte, Yte);

fprintf('\n--- BGP jump process, d = %d, weights = %s ---\n', d, tag);
fprintf('RMSE (median)  : %.4f   (irreducible noise SD %.2f)\n', m.RMSE, noiseSD);
fprintf('CRPS           : %.4f\n', m.CRPS);
fprintf('90%% coverage   : %.3f  (nominal 0.900)\n', m.Coverage);
fprintf('90%% mean width : %.4f\n', m.Width);

%% ---------------------------------------------------------------- plots
figure('Name','GBC - 2D jump process','Position',[80 80 1200 800]);

% Truth on the design points.
subplot(2,3,1);
scatter(X(:,1), X(:,2), 14, f, 'filled');
colorbar; axis square; title('true f (jump across a''x = 0)');
xlabel('x_1'); ylabel('x_2');

% Predicted median on a grid.
ng = 90;
[g1,g2] = meshgrid(linspace(-0.5,0.5,ng));
Xg = [g1(:) g2(:)];
Qg = gbcPredict(model, Xg, [0.05 0.5 0.95]);

subplot(2,3,2);
imagesc(linspace(-0.5,0.5,ng), linspace(-0.5,0.5,ng), reshape(Qg(:,2),ng,ng));
set(gca,'YDir','normal'); colorbar; axis square;
title('GBC predictive median'); xlabel('x_1'); ylabel('x_2');

subplot(2,3,3);
imagesc(linspace(-0.5,0.5,ng), linspace(-0.5,0.5,ng), ...
        reshape(Qg(:,3)-Qg(:,1),ng,ng));
set(gca,'YDir','normal'); colorbar; axis square;
title('90% interval width (widens at the jump)');
xlabel('x_1'); ylabel('x_2');

% Predictive densities at three points: deep in each regime, and on the
% boundary where the conditional law should be bimodal.
distFromBoundary = (Xg*a.')/norm(a);
[~,iBoundary] = min(abs(distFromBoundary));
[~,iDeep1]    = max(distFromBoundary);
[~,iDeep2]    = min(distFromBoundary);

pts   = [iDeep1, iBoundary, iDeep2];
names = {'deep in regime 1','on the boundary','deep in regime 2'};
for k = 1:3
    subplot(2,3,3+k);
    s = gbcSample(model, Xg(pts(k),:), 4000);
    histogram(s, 45, 'Normalization','pdf', 'FaceColor',[0.35 0.55 0.75], ...
        'EdgeColor','none');
    title(names{k}); xlabel('y'); ylabel('density'); grid on;
end

%% ---------------------------------------------------------------- helpers
function D2 = pdist2sq(A,B)
%PDIST2SQ Pairwise squared Euclidean distances without a toolbox dependency.
D2 = sum(A.^2,2) + sum(B.^2,2).' - 2*(A*B.');
D2 = max(D2, 0);
end
