function S = gbcSample(model, Xnew, B)
%GBCSAMPLE Draw predictive samples from the GBC surrogate.
%
%   S = GBCSAMPLE(model,Xnew,B) implements the test phase of Algorithm 1:
%
%       for b = 1..B:  tau^(b) ~ U[0,1],  y^(b) = q_hat_{tau^(b)}(x*)
%
%   Each test point receives its own independent tau draws, so the columns of
%   S are exchangeable draws from the estimated conditional distribution
%   rather than a shared quantile path.
%
%   Inputs
%     model : struct from GBCTRAIN.
%     Xnew  : nNew-by-d matrix of test inputs.
%     B     : number of predictive draws per input (default 200).
%
%   Output
%     S : nNew-by-B matrix of predictive samples on the original scale.
%
%   Because a single forward pass yields a draw, generating a full predictive
%   distribution costs B network evaluations and no linear algebra - this is
%   the O(n) test-time behavior that motivates GBC over a GP surrogate.
%
%   See also GBCPREDICT, GBCTRAIN.

%   ENSEMBLES. If model is a cell array (see GBCENSEMBLE), each of the K
%   members draws B samples and the results are pooled, giving n-by-(K*B).

arguments
    model {mustBeA(model,["struct","cell","gbcModel"])}
    Xnew  (:,:) double
    B     (1,1) double {mustBePositive, mustBeInteger} = 200
end

if iscell(model)
    parts = cell(1,numel(model));
    for k = 1:numel(model)
        parts{k} = gbcSample(model{k}, Xnew, B);
    end
    S = [parts{:}];
    return
end

if size(Xnew,2) ~= model.dIn
    error('gbcSample:BadWidth','Model expects %d predictors, got %d.', ...
          model.dIn, size(Xnew,2));
end

n = size(Xnew,1);

% An independent tau per (point, draw), exactly as Algorithm 1's test phase
% specifies. Generating the whole matrix up front draws the same values in the
% same order as filling it block by block - MATLAB fills column-major - and it
% lets GBCEVALTAU own the batching of the forward pass.
tauMat = rand(n, B, 'single');

S = gbcEvalTau(model, Xnew, tauMat);
end
