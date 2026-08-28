function params = gbcInit(dIn, opts)
%GBCINIT Initialise the learnable parameters of the Implicit Quantile Network.
%
%   params = GBCINIT(dIn, opts) returns a struct of dlarray parameters for
%
%       h   = f_1( f_x(x) .* f_tau(phi(tau)) )        ReLU, width HiddenSize
%       out = f_out( tanh( f_2(h) ) )                 f_2 width BottleneckSize
%
%   f_out produces two values: the location anchor mu_hat and the quantile
%   estimate q_hat_tau.
%
%   The Tanh bottleneck (f_2) is present in the authors' reference
%   implementation but absent from the architecture equation printed in the
%   paper. Set opts.BottleneckSize = 0 to drop it and reproduce the paper's
%   literal description instead.
%
%   Weights use Xavier/Glorot-uniform initialisation and biases start at zero,
%   matching nn.init.xavier_uniform_ / nn.init.zeros_ in the reference.

h  = opts.HiddenSize;
nb = opts.BottleneckSize;
nh = opts.NumCosine;

params.Wx = glorot([h dIn]);   params.bx = zeros1(h);
params.Wt = glorot([h nh]);    params.bt = zeros1(h);
params.W1 = glorot([h h]);     params.b1 = zeros1(h);

if nb > 0
    params.W2 = glorot([nb h]); params.b2 = zeros1(nb);
    params.Wo = glorot([2 nb]); params.bo = zeros1(2);
else
    params.Wo = glorot([2 h]);  params.bo = zeros1(2);
end
end

% -------------------------------------------------------------------------
function W = glorot(sz)
fanOut = sz(1);
fanIn  = sz(2);
bound  = sqrt(6/(fanIn+fanOut));
W = dlarray( single( (rand(sz)*2 - 1) * bound ) );
end

function b = zeros1(n)
b = dlarray(zeros(n,1,'single'));
end
