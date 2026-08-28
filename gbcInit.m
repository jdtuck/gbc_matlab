function params = gbcInit(dIn, opts)
%GBCINIT Initialise the learnable parameters of the Implicit Quantile Network.
%
%   params = GBCINIT(dIn, opts) returns a struct of dlarray parameters for
%
%       G_phi(tau,x) = f_out( f_1( f_x(x) .* f_tau(phi(tau)) ) )
%
%   with f_x, f_tau, f_1 fully-connected + ReLU (width opts.HiddenSize) and
%   f_out a linear layer producing two values: the location anchor mu_hat and
%   the quantile estimate q_hat_tau.
%
%   Weights use Glorot-uniform initialisation; biases start at zero.

h  = opts.HiddenSize;
nh = opts.NumCosine;

params.Wx = glorot([h dIn]);   params.bx = dlarray(zeros(h,1,'single'));
params.Wt = glorot([h nh]);    params.bt = dlarray(zeros(h,1,'single'));
params.W1 = glorot([h h]);     params.b1 = dlarray(zeros(h,1,'single'));
params.Wo = glorot([2 h]);     params.bo = dlarray(zeros(2,1,'single'));
end

function W = glorot(sz)
fanOut = sz(1);
fanIn  = sz(2);
bound  = sqrt(6/(fanIn+fanOut));
W = dlarray( single( (rand(sz)*2 - 1) * bound ) );
end
