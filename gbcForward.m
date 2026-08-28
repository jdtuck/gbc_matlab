function [muHat, qHat] = gbcForward(params, X, Phi)
%GBCFORWARD Forward pass of the Implicit Quantile Network.
%
%   [muHat,qHat] = GBCFORWARD(params,X,Phi) evaluates
%
%       h   = f_1( f_x(x) .* f_tau(phi(tau)) )     fully-connected + ReLU
%       out = f_out( tanh( f_2(h) ) )              f_2 present iff W2 exists
%
%   Inputs
%     params : struct from GBCINIT.
%     X      : d-by-B dlarray, format 'CB'  (standardised predictors).
%     Phi    : nh-by-B dlarray, format 'CB' (cosine quantile embedding).
%              When one tau is shared by the whole batch, GBCTRAIN expands it
%              to B identical columns before calling this, rather than relying
%              on implicit expansion across a labelled dimension.
%
%   Outputs
%     muHat : 1-by-B location anchor (training-time regulariser only).
%     qHat  : 1-by-B estimate of the conditional quantile at level tau.
%
%   The elementwise product is the multiplicative "conditioning" merge used by
%   implicit quantile networks: the quantile branch modulates the feature
%   branch rather than being concatenated to it, which is what lets a single
%   network represent the whole quantile curve.

hx = relu(fullyconnect(X,   params.Wx, params.bx));
ht = relu(fullyconnect(Phi, params.Wt, params.bt));

hm = hx .* ht;                                   % Hadamard merge

h1 = relu(fullyconnect(hm, params.W1, params.b1));

if isfield(params,'W2')
    h1 = tanh(fullyconnect(h1, params.W2, params.b2));
end

out = fullyconnect(h1, params.Wo, params.bo);

muHat = out(1,:);
qHat  = out(2,:);
end
