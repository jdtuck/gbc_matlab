function Q = gbcEvalTau(model, Xnew, tauMat)
%GBCEVALTAU Evaluate the fitted quantile surface at explicit (point, tau) pairs.
%
%   Q = GBCEVALTAU(model,Xnew,tauMat) returns q_hat_{tau}(x) for every test
%   point and every requested quantile level, where
%
%     tauMat is 1-by-m : level j is SHARED by all n test points. Column j of Q
%                        is then one coherent draw of the quantile surface,
%                        evaluated at all n points.
%     tauMat is n-by-m : tauMat(i,j) is the level used for point i in column j,
%                        so the columns carry independent per-point levels.
%
%   This is the primitive GBCPREDICT, GBCSAMPLE and GBCMODEL/PREDICT are built
%   on: the three differ only in where tau comes from (a caller's grid, fresh
%   uniforms, a stored sample set), never in how the network is evaluated.
%
%   Inputs
%     model  : struct from GBCTRAIN, or a gbcModel.
%     Xnew   : nNew-by-d matrix of test inputs.
%     tauMat : 1-by-m or nNew-by-m matrix of levels in [0,1].
%
%   Output
%     Q : nNew-by-m matrix on the original response scale. Column order is
%         exactly the column order of tauMat - nothing is sorted here.
%
%   Levels are evaluated in BLOCKS of columns, tiling the test points across
%   the levels in the block, so one forward pass covers many (point, level)
%   pairs. Looping one level at a time would pay MATLAB's per-call dlarray
%   overhead m times over for no reason; the block size caps peak activation
%   memory at roughly HiddenSize * maxCols singles.
%
%   See also GBCPREDICT, GBCSAMPLE, GBCMODEL, GBCFORWARD.

arguments
    model  {mustBeA(model,["struct","gbcModel"])}
    Xnew   (:,:) double
    tauMat (:,:) double
end

if size(Xnew,2) ~= model.dIn
    error('gbcEvalTau:BadWidth','Model expects %d predictors, got %d.', ...
          model.dIn, size(Xnew,2));
end

n = size(Xnew,1);
m = size(tauMat,2);

shared = (size(tauMat,1) == 1);         % broadcast a level across all points
if ~shared && size(tauMat,1) ~= n
    error('gbcEvalTau:BadTauSize', ...
          'tauMat must have 1 or %d rows (one per test point), got %d.', ...
          n, size(tauMat,1));
end
if any(tauMat(:) < 0 | tauMat(:) > 1) || any(isnan(tauMat(:)))
    error('gbcEvalTau:BadTau','Quantile levels must lie in [0,1].');
end

if n == 0 || m == 0
    Q = zeros(n, m);
    return
end

nh    = model.opts.NumCosine;
onGPU = paramsOnGPU(model.params);

Xs = (Xnew - model.muX) ./ model.sdX;
X0 = single(Xs.');                        % d-by-n
if onGPU, X0 = gpuArray(X0); end

tauS    = single(tauMat);
maxCols = max(1000, round(5e6 / max(1, model.opts.HiddenSize)));
T       = max(1, min(m, floor(maxCols / max(1,n))));

Q = zeros(n, m);
for s = 1:T:m
    blk = s:min(s+T-1, m);
    nb  = numel(blk);

    % Column layout within a block: test point varies fastest, then level.
    % The shared case is expanded here rather than up front, so a wide grid
    % never materialises an n-by-m copy of one row of levels.
    Xrep = repmat(X0, 1, nb);                      % d-by-(n*nb)
    if shared
        tRep = repelem(tauS(blk), n);              % 1-by-(n*nb)
    else
        tRep = reshape(tauS(:,blk), 1, n*nb);
    end
    if onGPU, tRep = gpuArray(tRep); end

    Xd  = dlarray(Xrep, 'CB');
    Phi = dlarray(quantileEmbedding(tRep, nh), 'CB');

    [~, qHat] = gbcForward(model.params, Xd, Phi);

    Q(:,blk) = reshape(gather(double(extractdata(qHat))), n, nb);
end

Q = Q * model.sdY + model.muY;
end

% -------------------------------------------------------------------------
function tf = paramsOnGPU(params)
tf = false;
try
    tf = isgpuarray(extractdata(params.Wx));
catch
    try
        tf = isa(extractdata(params.Wx),'gpuArray');
    catch
        tf = false;
    end
end
end
