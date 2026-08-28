function [muHat, qHat] = gbcForward(params, X, Phi)
%GBCFORWARD Forward pass of the Implicit Quantile Network.
%
%   [muHat,qHat] = GBCFORWARD(params,X,Phi) evaluates
%
%       G_phi(tau,x) = f_out( f_1( f_x(x) .* f_tau(phi(tau)) ) )
%
%   Inputs
%     params : struct from GBCINIT.
%     X      : d-by-B dlarray, format 'CB'  (standardised predictors).
%     Phi    : nh-by-B dlarray, format 'CB' (cosine quantile embedding).
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

h1 = relu(fullyconnect(hm,  params.W1, params.b1));
out =     fullyconnect(h1,  params.Wo, params.bo);

muHat = out(1,:);
qHat  = out(2,:);
end
