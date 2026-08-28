function phi = quantileEmbedding(tau, nh)
%QUANTILEEMBEDDING Cosine basis embedding of the quantile level tau.
%
%   phi = QUANTILEEMBEDDING(tau, nh) implements
%
%       phi(tau) = [ cos(j*pi*tau) ]_{j=0}^{nh-1}   in R^{nh}
%
%   (Polson & Sokolov 2026, Sec. 3; nh = 32 in the paper).
%
%   Inputs
%     tau : 1-by-B vector of quantile levels in [0,1] (row vector).
%     nh  : embedding size.
%
%   Output
%     phi : nh-by-B matrix, same class as tau.
%
%   The j = 0 term is the constant 1, which lets the tau-branch learn a
%   tau-independent gain in addition to the oscillatory components.

if ~isrow(tau)
    tau = reshape(tau,1,[]);
end
j   = cast((0:nh-1).', 'like', tau);
phi = cos(pi * (j * tau));
end
