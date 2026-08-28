function results = test_gbc(runSlow)
%TEST_GBC Verification suite for the MATLAB GBC / IQN implementation.
%
%   test_gbc          runs the fast checks (a few seconds).
%   test_gbc(true)    additionally runs the end-to-end calibration test,
%                     which trains a small network (~1-2 minutes on CPU).
%
%   Every check is against something independent of the implementation:
%   a closed-form value, a brute-force evaluation, finite differences, or an
%   analytically known conditional distribution.

if nargin < 1, runSlow = false; end

addpath(fileparts(fileparts(mfilename('fullpath'))));
rng(42);

state = struct('pass',0,'fail',0,'names',{{}},'msgs',{{}});

fprintf('=== GBC verification suite ===\n\n');

state = check(state, 'cosine embedding: shape and known values', @t_embedding);
state = check(state, 'CRPS: sorted-sum identity vs brute force',  @t_crps_identity);
state = check(state, 'CRPS: Gaussian grid vs closed form',        @t_crps_gaussian);
state = check(state, 'CRPS: permutation invariance',              @t_perm);
state = check(state, 'rearrangement: never raises the check loss',@t_rearrange);
state = check(state, 'pinball loss: minimiser is the quantile',   @t_pinball);
state = check(state, 'ordering surrogate: correct sign per tau',  @t_ordering);
state = check(state, 'loss weights: components isolate cleanly',  @t_weights);
state = check(state, 'gradients: dlgradient vs finite differences',@t_gradcheck);
state = check(state, 'forward pass: shapes and tau-dependence',   @t_forward);
state = check(state, 'predict: column order preserved',           @t_colorder);
state = check(state, 'train: reproducible under a fixed seed',    @t_seed);

if runSlow
    state = check(state, 'end-to-end: recovers analytic quantiles', @t_calibration);
else
    fprintf('  (skipping the end-to-end calibration test; run test_gbc(true) for it)\n');
end

fprintf('\n=== %d passed, %d failed ===\n', state.pass, state.fail);
results = state;
end

% =========================================================================
% harness
% =========================================================================
function state = check(state, name, fn)
try
    msg = fn();
    state.pass = state.pass + 1;
    fprintf('  PASS  %-48s %s\n', name, msg);
catch err
    state.fail = state.fail + 1;
    state.names{end+1} = name;
    state.msgs{end+1}  = err.message;
    fprintf('  FAIL  %-48s %s\n', name, err.message);
end
end

function assertTrue(cond, msg, varargin)
if ~all(cond(:))
    error('gbcTest:Failed', msg, varargin{:});
end
end

% =========================================================================
% tests
% =========================================================================
function msg = t_embedding()
nh  = 32;
tau = [0 0.25 0.5 1];
phi = quantileEmbedding(tau, nh);

assertTrue(isequal(size(phi),[nh numel(tau)]), 'embedding has the wrong size');

% brute force
ref = zeros(nh, numel(tau));
for j = 0:nh-1
    for k = 1:numel(tau)
        ref(j+1,k) = cos(j*pi*tau(k));
    end
end
e = max(abs(phi(:) - ref(:)));
assertTrue(e < 1e-12, 'embedding differs from the definition by %.3g', e);

% tau = 0 must give the all-ones vector; j = 0 row is always 1
assertTrue(all(abs(phi(:,1)-1) < 1e-12), 'phi(0) is not all ones');
assertTrue(all(abs(phi(1,:)-1) < 1e-12), 'the j=0 row is not constant');

msg = sprintf('max err %.1e', e);
end

% -------------------------------------------------------------------------
function msg = t_crps_identity()
worst = 0;
for M = [2 5 17 100]
    q = randn(1,M);
    brute = sum(sum(abs(q(:) - q(:).')))/(2*M^2);
    qs = sort(q);
    i  = 1:M;
    fast = sum((2*i - M - 1).*qs)/M^2;
    worst = max(worst, abs(brute-fast));
end
assertTrue(worst < 1e-11, 'pairwise term differs from brute force by %.3g', worst);

% and the full score against a fully brute-force CRPS
q = randn(1,40); y = 0.3;
brute = mean(abs(q-y)) - sum(sum(abs(q(:)-q(:).')))/(2*40^2);
assertTrue(abs(gbcCRPS(q,y) - brute) < 1e-11, 'gbcCRPS disagrees with brute force');

msg = sprintf('max err %.1e', worst);
end

% -------------------------------------------------------------------------
function msg = t_crps_gaussian()
% Closed form: CRPS(N(mu,sig), y) = sig*( z(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) )
mu = 1.3; sig = 0.7; y = 2.0;
z  = (y-mu)/sig;
closed = sig*( z*(2*ncdf(z)-1) + 2*npdf(z) - 1/sqrt(pi) );

M   = 4000;
tau = (1:M)/(M+1);
q   = mu + sig*sqrt(2)*erfinv(2*tau-1);      % Gaussian inverse CDF
est = gbcCRPS(q, y);

relErr = abs(est-closed)/closed;
assertTrue(relErr < 2e-3, 'grid CRPS off by %.2f%% (est %.6f, closed %.6f)', ...
           100*relErr, est, closed);

% coarser grids should still be close, and converge monotonically toward it
errs = zeros(1,4); Ms = [50 200 800 3200];
for k = 1:4
    t = (1:Ms(k))/(Ms(k)+1);
    errs(k) = abs(gbcCRPS(mu + sig*sqrt(2)*erfinv(2*t-1), y) - closed);
end
assertTrue(errs(end) < errs(1)/8, ...
    'CRPS error does not shrink with M: %s', mat2str(errs,3));

msg = sprintf('rel err %.2e at M=%d', relErr, M);
end

% -------------------------------------------------------------------------
function msg = t_perm()
% The CRPS estimator is a symmetric function of the quantile set, so the
% internal sort in gbcCRPS cannot change the score it reports.
worst = 0;
for r = 1:200
    q = randn(1,25);
    y = randn;
    worst = max(worst, abs(gbcCRPS(q,y) - gbcCRPS(q(randperm(25)),y)));
end
assertTrue(worst < 1e-12, 'CRPS is not permutation invariant (%.3g)', worst);
msg = sprintf('max diff %.1e over 200 draws', worst);
end

% -------------------------------------------------------------------------
function msg = t_rearrange()
% Chernozhukov-Fernandez-Galichon: sorting a crossing quantile curve weakly
% reduces its aggregate check-loss risk. This is what justifies the default
% rearrangement in gbcPredict.
worstIncrease = -inf;
for r = 1:300
    M   = randi([3 30]);
    q   = randn(1,M)*(0.2+1.8*rand);
    tau = sort(rand(1,M));
    y   = randn(1,200)*(0.2+1.8*rand) + randn;
    worstIncrease = max(worstIncrease, checkRisk(sort(q),tau,y) - checkRisk(q,tau,y));
end
assertTrue(worstIncrease < 1e-12, 'rearrangement raised the check loss by %.3g', ...
           worstIncrease);
msg = sprintf('max increase %.1e over 300 draws', max(worstIncrease,0));
end

% -------------------------------------------------------------------------
function msg = t_pinball()
% The population minimiser of the check loss is the tau-quantile. Verify the
% sample version: argmin over a fine grid must land on the empirical quantile.
y  = randn(4000,1);
ys = sort(y);
grid = linspace(-4,4,8001);
worst = 0;
for tau = [0.1 0.25 0.5 0.75 0.9]
    e = y.' - grid.';                                  % nGrid-by-n
    L = mean(max(tau*e, (tau-1)*e), 2);
    [~,k] = min(L);
    worst = max(worst, abs(grid(k) - empQuantile(ys,tau)));
end
assertTrue(worst < 0.02, ...
    'pinball minimiser off the empirical quantile by %.3g', worst);
msg = sprintf('max deviation %.1e', worst);
end

% -------------------------------------------------------------------------
function msg = t_ordering()
% m_tau must penalise q > y below the median and q < y above it, and be zero
% on the correct side. Probe the loss with w = [0 1 0].
d = 3; opts = gbcOptions('HiddenSize',8,'NumCosine',4);
params = gbcInit(d, opts);

X   = dlarray(single(randn(d,6)),'CB');
y   = zeros(1,6,'single');
w   = [0 1 0];

% Force q_hat to a known constant by zeroing everything but the output bias.
params = zeroParams(params);
for c = [-1 1]
    params.bo(2) = c;
    Phi = dlarray(quantileEmbedding(single([0.1 0.2 0.3 0.7 0.8 0.9]),4),'CB');
    tau = single([0.1 0.2 0.3 0.7 0.8 0.9]);
    [~, parts] = gbcLossNoGrad(params, X, Phi, tau, y, w, 0);
    ord = parts(2);
    if c > 0
        % q_hat = +1 > y = 0: penalised only for tau < 0.5
        expect = mean([0.4 0.3 0.2].*1)/2;   % |tau-0.5| * 1, averaged over 6
        assertTrue(abs(ord-expect) < 1e-6, ...
            'ordering term for q>y is %.6f, expected %.6f', ord, expect);
    else
        expect = mean([0.2 0.3 0.4].*1)/2;
        assertTrue(abs(ord-expect) < 1e-6, ...
            'ordering term for q<y is %.6f, expected %.6f', ord, expect);
    end
end
msg = 'sign convention matches Eq. (1)';
end

% -------------------------------------------------------------------------
function msg = t_weights()
% The total loss must be exactly the weighted sum of the three reported parts.
d = 4; opts = gbcOptions('HiddenSize',16,'NumCosine',8);
params = gbcInit(d, opts);
X   = dlarray(single(randn(d,32)),'CB');
tau = rand(1,32,'single');
Phi = dlarray(quantileEmbedding(tau,8),'CB');
y   = single(randn(1,32));
w   = [0.1 0.2 0.7];

[L, ~, parts] = dlfeval(@gbcLoss, params, X, Phi, tau, y, w, 0);
L = double(extractdata(L));

recon = w*parts(:);
assertTrue(abs(L-recon) < 1e-6*max(1,abs(L)), ...
    'loss %.8f != weighted parts %.8f', L, recon);

% Independent reimplementation of the same arithmetic must agree.
Lref = double(extractdata(gbcLossNoGrad(params, X, Phi, tau, y, w, 0)));
assertTrue(abs(L-Lref) < 1e-6*max(1,abs(L)), ...
    'gbcLoss %.8f != reference implementation %.8f', L, Lref);

msg = sprintf('|loss - w''*parts| = %.1e', abs(L-recon));
end

% -------------------------------------------------------------------------
function msg = t_gradcheck()
% Central finite differences against dlgradient, in double precision.
%
% The loss is piecewise LINEAR in the parameters (ReLU units, and abs/max in
% all three loss terms), so away from a kink the central difference is exact
% to round-off and the tolerance below can be tight. On a kink it is
% meaningless: dlgradient takes a one-sided subgradient while the central
% difference averages the two sides.
%
% Kinks are not a measure-zero worry at the default initialisation. gbcInit
% sets every bias to exactly zero, so any example whose first-layer
% pre-activations are all negative gives hx = 0, hence hm = hx.*ht = 0, hence
% a third-layer pre-activation of exactly W1*0 + b1 = 0 - sitting precisely on
% the ReLU kink. With 6 units that is (1/2)^6 per example, so ~27% of 20-point
% batches contain one, and probing b1 there produces exactly the spurious
% mismatch this comment exists to prevent.
%
% So: give the biases nonzero values, and refuse to probe until every kink is
% far away compared with the step size h.

d = 3; nh = 5; hid = 6;
opts = gbcOptions('HiddenSize',hid,'NumCosine',nh);

h      = 1e-5;
margin = 1e-3;              % require every kink to be >> h away
params = [];
for attempt = 1:50
    p = gbcInit(d, opts);
    p = structfun(@(v) dlarray(double(extractdata(v))), p, 'UniformOutput', false);

    % Nonzero biases: this is what breaks the exact-zero degeneracy above.
    p.bx = dlarray(0.3*randn(hid,1));
    p.bt = dlarray(0.3*randn(hid,1));
    p.b1 = dlarray(0.3*randn(hid,1));
    p.bo = dlarray(0.3*randn(2,1));

    Xc   = dlarray(randn(d,20),'CB');
    tauc = 0.02 + 0.96*rand(1,20);
    Phic = dlarray(quantileEmbedding(tauc,nh),'CB');
    yc   = randn(1,20);

    if kinkMargin(p, Xc, Phic, yc) > margin
        params = p; X = Xc; tau = tauc; Phi = Phic; y = yc;
        break
    end
end
assertTrue(~isempty(params), ...
    'could not draw a probe point with all kinks more than %g away', margin);

w = [0.3 0.3 0.4];

[L0, grads] = dlfeval(@gbcLoss, params, X, Phi, tau, y, w, 0);

% The finite difference below resolves a change of order h*|grad| ~ 1e-6. That
% is invisible in single precision (~7 significant digits), so a loss that
% silently downcast to single would make every fd exactly zero and report a
% relative error of 1 with no hint as to why. Check the precision explicitly.
assertTrue(strcmp(underlyingType(L0),'double'), ...
    ['the loss came back as %s, not double: some operand is pinning the ' ...
     'precision (check for casts like single(...) in gbcLoss), which makes ' ...
     'the finite difference probe blind'], underlyingType(L0));

names = fieldnames(params);
worst = 0;
worstWhere = '';
sawNonzero = false;
for k = 1:numel(names)
    Pd = extractdata(params.(names{k}));
    g  = extractdata(grads.(names{k}));
    idxs = randperm(numel(Pd), min(4, numel(Pd)));
    for ii = idxs
        Pp = Pd; Pp(ii) = Pp(ii) + h;
        Pm = Pd; Pm(ii) = Pm(ii) - h;
        pp = params; pp.(names{k}) = dlarray(Pp);
        pm = params; pm.(names{k}) = dlarray(Pm);
        Lp = double(extractdata(gbcLossNoGrad(pp, X, Phi, tau, y, w, 0)));
        Lm = double(extractdata(gbcLossNoGrad(pm, X, Phi, tau, y, w, 0)));
        fd = (Lp-Lm)/(2*h);
        an = double(g(ii));

        % Both exactly zero is legitimate: a dead ReLU unit contributes no
        % gradient at all. Only a disagreement between them is a failure.
        if abs(fd) > 0 || abs(an) > 0
            sawNonzero = true;
        end

        rel = abs(fd-an)/max(1e-4, abs(fd)+abs(an));
        if rel > worst
            worst = rel;
            worstWhere = sprintf('%s(%d): fd %.6g vs analytic %.6g', ...
                                 names{k}, ii, fd, an);
        end
    end
end

assertTrue(sawNonzero, ...
    'every probed gradient was zero - the check would pass vacuously');
assertTrue(worst < 1e-4, 'worst relative gradient error %.3g at %s', ...
           worst, worstWhere);
msg = sprintf('worst rel err %.1e over %d probes', worst, 4*numel(names));
end

% -------------------------------------------------------------------------
function msg = t_forward()
d = 5; opts = gbcOptions('HiddenSize',32,'NumCosine',16);
params = gbcInit(d, opts);
X = dlarray(single(randn(d,7)),'CB');

tauA = repmat(single(0.1),1,7);
tauB = repmat(single(0.9),1,7);
[muA,qA] = gbcForward(params, X, dlarray(quantileEmbedding(tauA,16),'CB'));
[muB,qB] = gbcForward(params, X, dlarray(quantileEmbedding(tauB,16),'CB'));

assertTrue(isequal(size(qA),[1 7]), 'q head has the wrong shape');
assertTrue(isequal(size(muA),[1 7]), 'mu head has the wrong shape');

dq = max(abs(extractdata(qA-qB)));
assertTrue(dq > 1e-6, 'q_hat does not depend on tau (dq = %.3g)', dq);

% mu is also a function of tau at initialisation (it shares the trunk); what
% matters is that the two heads are distinct.
dHeads = max(abs(extractdata(muA-qA)));
assertTrue(dHeads > 1e-8, 'the two output heads are identical');

msg = sprintf('tau sensitivity %.2f', dq);
end

% -------------------------------------------------------------------------
function msg = t_colorder()
% gbcPredict must return columns in the caller's tau order, not sorted order.
X = rand(30,2); Y = X*[1;2] + 0.1*randn(30,1);
model = gbcTrain(X, Y, gbcOptions('MaxEpochs',5,'HiddenSize',16, ...
    'Verbose',false,'Seed',3));

tauA = [0.1 0.5 0.9];
tauB = [0.9 0.5 0.1];
QA = gbcPredict(model, X, tauA);
QB = gbcPredict(model, X, tauB);

e = max(abs(QA(:) - reshape(fliplr(QB),[],1)));
assertTrue(e < 1e-12, 'column order not preserved (max diff %.3g)', e);
assertTrue(all(all(diff(QA,1,2) >= -1e-12)), 'rearranged quantiles are not monotone');

msg = 'order preserved, output monotone in tau';
end

% -------------------------------------------------------------------------
function msg = t_seed()
X = rand(120,3); Y = sum(X,2) + 0.2*randn(120,1);
o = gbcOptions('MaxEpochs',10,'HiddenSize',24,'Verbose',false,'Seed',11);
m1 = gbcTrain(X,Y,o);
m2 = gbcTrain(X,Y,o);
Q1 = gbcPredict(m1, X(1:5,:), [0.25 0.75]);
Q2 = gbcPredict(m2, X(1:5,:), [0.25 0.75]);
e = max(abs(Q1(:)-Q2(:)));
assertTrue(e < 1e-10, 'runs with the same seed differ by %.3g', e);
msg = sprintf('max diff %.1e', e);
end

% -------------------------------------------------------------------------
function msg = t_calibration()
% Heteroskedastic 1-D problem with analytically known conditional quantiles:
%     y | x ~ N( sin(2*pi*x),  (0.1 + 0.4*x)^2 ),   x ~ U[0,1]
% The fitted quantile curve must track the analytic one, and 90% intervals
% must cover close to 90%.
rng(5);
n = 3000;
x = rand(n,1);
mu = @(x) sin(2*pi*x);
sd = @(x) 0.1 + 0.4*x;
y = mu(x) + sd(x).*randn(n,1);

model = gbcTrain(x, y, gbcOptions('MaxEpochs',1200,'MiniBatchSize',256, ...
    'Verbose',false,'Seed',5));

xs   = (0.05:0.05:0.95).';
taus = [0.1 0.25 0.5 0.75 0.9];
Q    = gbcPredict(model, xs, taus);
Qtrue = mu(xs) + sd(xs).*(sqrt(2)*erfinv(2*taus-1));

absErr  = abs(Q - Qtrue);
scale   = mean(sd(xs));
meanErr = mean(absErr(:));
maxErr  = max(absErr(:));
assertTrue(meanErr < 0.30*scale, ...
    'mean quantile error %.3f is %.0f%% of the mean noise SD', meanErr, 100*meanErr/scale);
assertTrue(maxErr < 1.00*scale, ...
    'worst quantile error %.3f exceeds the mean noise SD %.3f', maxErr, scale);
err = maxErr;

xt = rand(4000,1);
yt = mu(xt) + sd(xt).*randn(4000,1);
m  = gbcMetrics(model, xt, yt);
assertTrue(abs(m.Coverage-0.90) < 0.05, ...
    '90%% coverage is %.3f, outside [0.85,0.95]', m.Coverage);

msg = sprintf('max quantile err %.3f, coverage %.3f, CRPS %.3f', ...
              err, m.Coverage, m.CRPS);
end

% =========================================================================
% helpers
% =========================================================================
function [loss, parts] = gbcLossNoGrad(params, X, Phi, tau, Y, w, l2)
%GBCLOSSNOGRAD Same arithmetic as gbcLoss but without the dlgradient call,
%   so the loss value can be probed outside a traced dlfeval context. Written
%   independently of gbcLoss on purpose: t_weights compares the two.
[muHat, qHat] = gbcForward(params, X, Phi);
e = Y - qHat;
lAnchor  = sum(abs(Y-muHat),2)/numel(Y);
isLow    = tau < 0.5;
mTau     = isLow.*max(0,qHat-Y) + (~isLow).*max(0,Y-qHat);
lOrder   = sum(abs(tau-0.5).*mTau,2)/numel(Y);
lPinball = sum(max(tau.*e,(tau-1).*e),2)/numel(Y);
loss = w(1)*lAnchor + w(2)*lOrder + w(3)*lPinball;
if l2 > 0
    loss = loss + l2*(sum(params.Wx.^2,'all')+sum(params.Wt.^2,'all')+ ...
                      sum(params.W1.^2,'all')+sum(params.Wo.^2,'all'));
end
parts = [double(extractdata(lAnchor)), double(extractdata(lOrder)), ...
         double(extractdata(lPinball))];
end

function m = kinkMargin(params, X, Phi, Y)
%KINKMARGIN Distance from the current probe point to the nearest kink of the
%   loss surface. The loss is piecewise linear, and every kink comes from one
%   of five places: the three ReLU pre-activations, the L1 anchor's |y - mu|,
%   and the residual y - q that both the pinball loss and the ordering
%   surrogate hinge on. A central finite difference is only valid if it does
%   not straddle any of them, so the gradient check refuses to run until this
%   margin comfortably exceeds the step size.
zx = fullyconnect(X,   params.Wx, params.bx);
zt = fullyconnect(Phi, params.Wt, params.bt);
hm = relu(zx) .* relu(zt);
z1 = fullyconnect(hm,  params.W1, params.b1);
out = fullyconnect(relu(z1), params.Wo, params.bo);

resid  = Y - out(2,:);          % pinball and ordering kinks
anchor = Y - out(1,:);          % L1 anchor kink

% Unwrap before linear indexing: (:) is not defined on a formatted dlarray.
ezx = extractdata(zx);  ezt = extractdata(zt);  ez1 = extractdata(z1);
er  = extractdata(resid); ea = extractdata(anchor);

vals = [abs(ezx(:)); abs(ezt(:)); abs(ez1(:)); abs(er(:)); abs(ea(:))];
m = double(min(vals));
end

function p = zeroParams(p)
f = fieldnames(p);
for k = 1:numel(f)
    p.(f{k}) = dlarray(zeros(size(p.(f{k})),'single'));
end
end

function r = checkRisk(q, tau, y)
%CHECKRISK Aggregate pinball loss of a quantile curve q at levels tau,
%   evaluated against the sample y.
e = y - q(:);                        % M-by-n
r = sum(mean(max(tau(:).*e, (tau(:)-1).*e), 2));
end

function v = empQuantile(ysorted, tau)
%EMPQUANTILE Lower empirical quantile, without the Statistics Toolbox.
n = numel(ysorted);
v = ysorted(min(n, max(1, ceil(tau*n))));
end

function p = ncdf(z), p = 0.5*erfc(-z/sqrt(2)); end
function p = npdf(z), p = exp(-0.5*z.^2)/sqrt(2*pi); end
