function T = bench_gbc(nList, d)
%BENCH_GBC Decompose the cost of one GBC training step, CPU and GPU.
%
%   bench_gbc                profiles n = [106 2000 20000]
%   bench_gbc(nList, d)      profiles the sizes you name at input dim d
%   T = bench_gbc(...)       also returns the measurements as a table
%
%   Pass n = 90000 to profile the Michalewicz working set (needs a few GB;
%   the autodiff tape holds several 256-by-n single activations).
%
%   WHY THIS EXISTS
%   The interesting question about any optimisation - MEX, a hand-coded
%   backward pass, batching, parfor, GPU - is whether a training step is
%   dominated by ARITHMETIC or by OVERHEAD.
%
%     - Arithmetic means BLAS matrix multiplies. MATLAB already dispatches
%       those to a tuned multithreaded BLAS. A MEX file calls the SAME
%       library, so it cannot make them faster. The only wins are doing fewer
%       FLOPs, or running them on hardware with more of them.
%
%     - Overhead means everything else: the autodiff tape built inside
%       dlfeval, dlarray dispatch per elementwise op, the per-tensor loops in
%       adamupdate and dlupdate. A hand-coded backward pass or a MEX file can
%       remove this - and only this.
%
%   Overhead is a fixed cost per step while arithmetic grows with n, so the
%   ratio below should fall sharply as n grows. That is the whole answer to
%   "should we MEX this": it depends on which n you care about.
%
%   Interpreting the ratio (measured step / BLAS floor):
%       < 1.5x   compute-bound. MEX buys ~nothing. Use a GPU, or fewer FLOPs.
%       1.5-3x   mixed. A hand-coded backward pass captures most of the gap.
%       > 3x     overhead-bound. Hand-code first; then re-measure.

arguments
    nList (1,:) double {mustBePositive, mustBeInteger} = [106 2000 20000]
    d     (1,1) double {mustBePositive, mustBeInteger} = 4
end

here = fileparts(mfilename('fullpath'));
addpath(here);

H  = 256;    % HiddenSize
Bn = 64;     % BottleneckSize
nh = 32;     % NumCosine

opts = gbcOptions('HiddenSize',H,'BottleneckSize',Bn,'NumCosine',nh, ...
                  'Verbose',false);

useGPU = false;
gpuName = 'none';
try
    if gpuDeviceCount("available") > 0
        g = gpuDevice;
        useGPU = true;
        gpuName = sprintf('%s, %.1f GB', g.Name, g.TotalMemory/1e9);
    end
catch
end

fprintf('\n=== GBC step-cost profile ===\n');
fprintf('MATLAB %s | %d cores | GPU: %s\n', ...
        version('-release'), feature('numcores'), gpuName);
fprintf('d=%d hidden=%d bottleneck=%d cosine=%d\n\n', d, H, Bn, nh);

% ---- warm up every timed code path before measuring anything ------------
% Without this the FIRST n in the sweep absorbs one-time costs - JIT, dlarray
% class loading, deep-learning library init - and reports a wildly inflated
% overhead ratio. timeit's own warmup does not cover library initialisation.
%
% The self-check below catches any residue: dlupdate and adamupdate touch only
% the parameters, so their cost is independent of n. If that column is not
% flat across the sweep, warmup leaked in and the small-n rows are not real.
wp = gbcInit(d, opts);
wX = dlarray(randn(d,64,'single'),'CB');
wT = rand(1,64,'single');
wPhi = dlarray(quantileEmbedding(wT,nh),'CB');
wY = randn(1,64,'single');
wPlain = structfun(@extractdata, wp, 'UniformOutput', false);
for rep = 1:3
    [~, wg] = dlfeval(@gbcLoss, wp, wX, wPhi, wT, wY, [0.3 0.3 0.4], 0);
    dlupdate(@(g,p) g + 1e-4*p, wg, wp);
    za = []; zb = [];
    adamupdate(wp, wg, za, zb, 1, 1e-3, 0.9, 0.999);
    plainForward(wPlain, extractdata(wX), extractdata(wPhi));
    wp.Wx*1;                                     %#ok<VUNUS> touch BLAS
end
clear wp wX wT wPhi wY wPlain wg za zb

rows = cell(numel(nList),1);

for k = 1:numel(nList)
    n = nList(k);

    params = gbcInit(d, opts);
    Xd   = dlarray(randn(d,n,'single'),'CB');
    tau  = rand(1,n,'single');
    Phi  = dlarray(quantileEmbedding(tau,nh),'CB');
    Y    = randn(1,n,'single');
    w    = [0.3 0.3 0.4];

    % ---- what we pay per step on CPU ------------------------------------
    tStep = timeit(@() dlfeval(@gbcLoss, params, Xd, Phi, tau, Y, w, 0), 2);

    [~, grads] = dlfeval(@gbcLoss, params, Xd, Phi, tau, Y, w, 0);
    tWd   = timeit(@() dlupdate(@(g,p) g + 1e-4*p, grads, params), 1);
    aG = []; aS = [];
    tAdam = timeit(@() adamupdate(params, grads, aG, aS, 1, 1e-3, 0.9, 0.999), 3);

    % ---- floor: same arithmetic, plain arrays, no tape ------------------
    P  = structfun(@extractdata, params, 'UniformOutput', false);
    tPlainFwd = timeit(@() plainForward(P, extractdata(Xd), extractdata(Phi)), 1);

    hm = randn(H, n, 'single');
    tG = timeit(@() P.W1*hm, 1);
    gflops = 2*H*H*n / tG / 1e9;

    fwdFlops  = 2*n*(H*d + H*nh + H*H + Bn*H + 2*Bn);
    stepFlops = 3*fwdFlops;
    blasFloor = stepFlops / (gflops*1e9);
    ratio     = tStep / blasFloor;
    perStep   = tStep + tWd + tAdam;

    % The idealised floor above assumes EVERY flop runs at the rate of the
    % big 256x256 GEMM. It does not: f_tau has K=32, f_out is tiny, and the
    % relu/tanh/product traffic is memory-bound, not flop-bound. So that
    % number badly overstates the opportunity.
    %
    % The achievable floor is what real hand-written code reaches: the
    % measured plain-array forward, times ~3 for the backward pass (backward
    % is ~2x the forward flops at similar memory behaviour), plus a fused
    % Adam. THIS is the number that says whether hand-coding or MEX is worth
    % writing.
    achFloor = 3*tPlainFwd + 0.15e-3;
    headroom = perStep / achFloor;

    % ---- the same step with the trace cached -----------------------------
    % This is the decisive measurement. If the step time is dominated by a
    % fixed tracing cost, caching the trace removes most of it; if it is
    % dominated by arithmetic, this changes nothing.
    tauD = dlarray(tau,'CB');
    tAcc = NaN;
    if ~isempty(which('dlaccelerate'))
        try
            af = dlaccelerate(@gbcLoss);
            dlfeval(af, params, Xd, Phi, tauD, Y, w, 0);   % populate cache
            tAcc = timeit(@() dlfeval(af, params, Xd, Phi, tauD, Y, w, 0), 2);
        catch err
            fprintf('   (dlaccelerate skipped: %s)\n', err.message);
        end
    end

    % ---- same step on the GPU -------------------------------------------
    tStepG = NaN; gflopsG = NaN; tAccG = NaN;
    if useGPU
        try
            pG   = dlupdate(@gpuArray, params);
            XdG  = dlarray(gpuArray(extractdata(Xd)),'CB');
            PhiG = dlarray(gpuArray(extractdata(Phi)),'CB');
            tauG = gpuArray(tau);
            YG   = gpuArray(Y);
            tauGD = dlarray(tauG,'CB');
            tStepG = gputimeit(@() dlfeval(@gbcLoss, pG, XdG, PhiG, tauGD, YG, w, 0), 2);
            gflopsG = stepFlops / tStepG / 1e9;

            if ~isempty(which('dlaccelerate'))
                afG = dlaccelerate(@gbcLoss);
                dlfeval(afG, pG, XdG, PhiG, tauGD, YG, w, 0);
                tAccG = gputimeit(@() dlfeval(afG, pG, XdG, PhiG, tauGD, YG, w, 0), 2);
            end
        catch err
            fprintf('   (GPU timing skipped: %s)\n', err.message);
        end
    end

    fprintf('n = %-6d  CPU BLAS %.1f GFLOPS | step %.2f MFLOP\n', ...
            n, gflops, stepFlops/1e6);
    fprintf('   fwd+bwd (dlfeval)      %9.3f ms\n', 1e3*tStep);
    fprintf('   forward (plain arrays) %9.3f ms   <- hand-coded floor\n', 1e3*tPlainFwd);
    fprintf('   weight decay + Adam    %9.3f ms\n', 1e3*(tWd+tAdam));
    fprintf('   ---------------------------------\n');
    fprintf('   per step (CPU)         %9.3f ms\n', 1e3*perStep);
    fprintf('   idealised floor        %9.3f ms  (%.2fx) <- all-flops-at-peak, optimistic\n', ...
            1e3*blasFloor, ratio);
    fprintf('   ACHIEVABLE floor       %9.3f ms\n', 1e3*achFloor);
    fprintf('   REAL HEADROOM          %9.2fx  <- what hand-coding could win\n', headroom);
    if ~isnan(tAcc)
        fprintf('   fwd+bwd ACCELERATED    %9.3f ms   %.2fx vs untraced\n', ...
                1e3*tAcc, tStep/tAcc);
    end
    if ~isnan(tStepG)
        fprintf('   per step (GPU)         %9.3f ms   %.1f GFLOPS, %.1fx vs CPU\n', ...
                1e3*tStepG, gflopsG, tStep/tStepG);
    end
    if ~isnan(tAccG)
        fprintf('   GPU + ACCELERATED      %9.3f ms   %.2fx vs GPU alone\n', ...
                1e3*tAccG, tStepG/tAccG);
    end
    fprintf('\n');

    rows{k} = {n, gflops, 1e3*tStep, 1e3*tPlainFwd, 1e3*(tWd+tAdam), ...
               1e3*perStep, 1e3*blasFloor, ratio, 1e3*achFloor, headroom, ...
               1e3*tAcc, 1e3*tStepG, gflopsG, 1e3*tAccG};
end

T = cell2table(vertcat(rows{:}), 'VariableNames', ...
    {'n','cpu_GFLOPS','fwdbwd_ms','fwd_plain_ms','opt_ms','step_ms', ...
     'idealised_floor_ms','idealised_ratio','achievable_floor_ms','headroom', ...
     'accel_ms','gpu_step_ms','gpu_GFLOPS','gpu_accel_ms'});

% ---- what this means for the large-n benchmarks -------------------------
fprintf('=== projected single-model training time (3000 steps) ===\n');
for k = 1:numel(nList)
    sCPU = 3000*T.step_ms(k)/1e3;
    if ~isnan(T.gpu_step_ms(k))
        sGPU = 3000*T.gpu_step_ms(k)/1e3;
        fprintf('  n=%-6d  CPU %8.1f s (%5.1f min)   GPU %8.1f s (%5.1f min)\n', ...
                T.n(k), sCPU, sCPU/60, sGPU, sGPU/60);
    else
        fprintf('  n=%-6d  CPU %8.1f s (%5.1f min)\n', T.n(k), sCPU, sCPU/60);
    end
end

% ---- the scaling trap in the reference protocol -------------------------
fprintf('\n=== FLOPs spent per quantile level visited ===\n');
fprintf('Under the reference protocol (full batch, ONE tau shared per step) a\n');
fprintf('step costs O(n) and yields exactly one quantile level. Per-example tau\n');
fprintf('pairs a distinct level with every training point, so (tau,x) diversity\n');
fprintf('per FLOP is higher by a factor of n:\n\n');
for k = 1:numel(nList)
    fprintf('  n=%-6d  reference: 1 level per %.1f MFLOP   |  per-example: %d levels for the same cost\n', ...
            T.n(k), 3*2*T.n(k)*(H*d+H*nh+H*H+Bn*H+2*Bn)/1e6, T.n(k));
end
fprintf('\nCaveat: these are not equivalent information. A shared tau pairs one\n');
fprintf('level with all n points; per-example tau pairs each level with a single\n');
fprintf('point. The diversity ratio is an upper bound on the benefit, not a\n');
fprintf('measured speedup - but at n=90,000 the reference spends ~49 GFLOP to\n');
fprintf('learn about one quantile level, which is worth knowing before scaling.\n');

% ---- verdict -------------------------------------------------------------
[~, iBig] = max(T.n);
hBig = T.headroom(iBig);
fprintf('\n=== verdict for the largest n profiled (%d) ===\n', T.n(iBig));
if hBig < 1.3
    fprintf('AT THE FLOOR (%.2fx headroom). The existing dlarray code is already\n', hBig);
    fprintf('within %.0f%% of what hand-written plain-array code could reach. Neither\n', 100*(hBig-1));
    fprintf('a hand-coded backward pass nor MEX is worth writing at this size -\n');
    fprintf('MEX would call the same BLAS and hit the same memory bandwidth.\n');
    fprintf('To go faster here you must do FEWER FLOPS, or find more hardware.\n');
elseif hBig < 2.5
    fprintf('MODEST HEADROOM (%.2fx). A hand-coded backward pass is worth it if\n', hBig);
    fprintf('this size dominates your runtime; MEX would add little beyond it.\n');
else
    fprintf('LARGE HEADROOM (%.2fx). Framework overhead dominates even at this\n', hBig);
    fprintf('size. Hand-code the backward pass in MATLAB first, then re-measure.\n');
end

if numel(nList) > 1
    fprintf('\nReal headroom across n: %s\n', ...
        strjoin(compose('n=%d -> %.2fx', T.n, T.headroom)', ', '));
    fprintf('Overhead is a fixed per-step cost, so it matters only at small n.\n');
end

% ---- parfor advice, which depends on WHICH regime you are in ------------
hasPCT = license('test','Distrib_Computing_Toolbox');
nc = feature('numcores');
fprintf('\nParallel Computing Toolbox: %s | %d cores\n', ...
        ternary(hasPCT,'available','NOT available'), nc);
if hasPCT
    hSmall = T.headroom(1);
    if hSmall > 3
        fprintf('At small n the step is overhead-bound, i.e. single-threaded MATLAB\n');
        fprintf('work while the cores sit idle - so parfor across replicates should\n');
        fprintf('scale close to %dx there.\n', nc);
    end
    if hBig < 1.5
        fprintf('At n=%d BLAS is ALREADY using every core (%.0f GFLOPS measured), so\n', ...
                T.n(iBig), T.cpu_GFLOPS(iBig));
        fprintf('parfor there would just split the same cores - expect little or no\n');
        fprintf('gain, and possible slowdown from oversubscription. Use parfor for\n');
        fprintf('the many-small-fits workloads, not the large-n ones.\n');
    end
end
end

% =========================================================================
function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end

function out = plainForward(P, X, Phi)
%PLAINFORWARD Same arithmetic as gbcForward on plain single arrays: no
%   dlarray, no tape. The floor a hand-coded forward pass would hit.
hx = max(0, P.Wx*X   + P.bx);
ht = max(0, P.Wt*Phi + P.bt);
h1 = max(0, P.W1*(hx.*ht) + P.b1);
if isfield(P,'W2')
    h1 = tanh(P.W2*h1 + P.b2);
end
out = P.Wo*h1 + P.bo;
end
