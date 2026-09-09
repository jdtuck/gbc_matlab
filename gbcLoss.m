function [loss, grads, parts] = gbcLoss(params, X, Phi, tau, Y, w, l2)
%GBCLOSS Three-term composite loss of Polson & Sokolov (2026), Eq. (1).
%
%       l(tau) = w1 * E|y - mu_hat|
%              + w2 * E[ |tau - 0.5| * m_tau ]
%              + w3 * E[ max(tau*e, (tau-1)*e) ],      e = y - q_hat_tau
%
%   with the ordering surrogate
%
%       m_tau = max(0, q_hat_tau - y)   if tau <  0.5
%               max(0, y - q_hat_tau)   if tau >= 0.5
%
%   Term 1 is an L1 anchor on the conditional median that suppresses mode
%   collapse; term 2 penalises quantile crossings (a low quantile above the
%   observation, or a high quantile below it), weighted by distance from the
%   median so the penalty bites hardest in the tails; term 3 is the standard
%   pinball / check loss whose population minimiser is the true conditional
%   quantile Q_tau(Y|x).
%
%   Inputs
%     params : struct of dlarray parameters (see GBCINIT).
%     X      : d-by-B  dlarray 'CB', standardised predictors.
%     Phi    : nh-by-B dlarray 'CB', cosine embedding of tau.
%     tau    : 1-by-B  numeric (untraced), one draw per training example.
%     Y      : 1-by-B  numeric or dlarray, standardised responses.
%     w      : [w1 w2 w3] loss weights.
%     l2     : optional L2 penalty on weight matrices (biases excluded).
%
%   Outputs
%     loss  : scalar dlarray.
%     grads : struct of gradients matching params (traced calls only).
%     parts : [anchor ordering pinball] components, as doubles.

if nargin < 7 || isempty(l2), l2 = 0; end

[muHat, qHat] = gbcForward(params, X, Phi);

e = Y - qHat;

% --- term 1: L1 location anchor -------------------------------------------
lAnchor = meanAll(abs(Y - muHat));

% --- term 2: quantile-ordering surrogate ----------------------------------
% Written branch-free. The masked form
%
%     m_tau  = max(0, q - y)  if tau <  0.5
%              max(0, y - q)  if tau >= 0.5
%     term   = |tau - 0.5| * m_tau
%
% is EXACTLY max(0, (tau - 0.5)*e) with e = y - q. For tau >= 0.5 the factor
% (tau - 0.5) is |tau - 0.5| and max(0, c*e) = c*max(0, e); for tau < 0.5 it
% is -|tau - 0.5|, so max(0, c*e) = |c|*max(0, -e) = |tau-0.5|*max(0, q-y).
% At tau = 0.5 both are zero. Verified bit-identical over random inputs and
% at the boundary cases (see t_ordering in test_gbc).
%
% This is not cosmetic. The masked version compared tau against a constant,
% and a comparison is frozen at trace time: under dlaccelerate the cached
% trace would keep the FIRST step's mask and silently apply it to every
% later tau. Here tau enters through arithmetic only, so the trace stays
% valid for any tau and the function is safe to accelerate.
lOrder = meanAll(max(0, (tau - 0.5) .* e));

% --- term 3: pinball / check loss -----------------------------------------
lPinball = meanAll(max(tau .* e, (tau - 1) .* e));

loss = w(1)*lAnchor + w(2)*lOrder + w(3)*lPinball;

if l2 > 0
    reg = sum(params.Wx.^2,'all') + sum(params.Wt.^2,'all') + ...
          sum(params.W1.^2,'all') + sum(params.Wo.^2,'all');
    loss = loss + l2*reg;
end

if nargout > 1
    grads = dlgradient(loss, params);
end
if nargout > 2
    parts = [gather(double(extractdata(lAnchor))), ...
             gather(double(extractdata(lOrder))),  ...
             gather(double(extractdata(lPinball)))];
end
end

% -------------------------------------------------------------------------
function m = meanAll(v)
%MEANALL Batch mean of a 1-by-B row. Reducing along the explicit dimension
%   keeps this valid for both formatted ('CB') and unformatted dlarrays.
m = sum(v,2) ./ size(v,2);
end
