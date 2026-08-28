function v = gbcRowQuantile(Ss, p)
%GBCROWQUANTILE Empirical quantile of each row of an already-sorted matrix.
%
%   v = GBCROWQUANTILE(Ss, p) returns the p-quantile of each row of Ss, which
%   must already be sorted ascending along dimension 2.
%
%   Uses linear interpolation at position p*(B-1)+1, which is numpy's default
%   ('linear') rule and therefore what the reference implementation's
%   np.quantile calls produce. Avoids a Statistics Toolbox dependency.
%
%   Beware what this does to a sample built from a quantile GRID rather than
%   from random draws. If the grid spans [a,b] instead of [0,1], the empirical
%   p-quantile lands at level a + p*(b-a), not p. That distortion is affine
%   and does NOT shrink as the grid is refined. See GBCQUANTILE and the
%   tauSpan argument of GBCMETRICSFROMSAMPLES, which undo it.

arguments
    Ss (:,:) double
    p  (1,1) double
end

B    = size(Ss,2);
pos  = p*(B-1) + 1;
loI  = max(1, min(B, floor(pos)));
hiI  = min(B, loI+1);
frac = pos - loI;

v = (1-frac).*Ss(:,loI) + frac.*Ss(:,hiI);
end
