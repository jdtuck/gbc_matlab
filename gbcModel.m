classdef gbcModel
    %GBCMODEL Fitted GBC / IQN surrogate, packaged for calibration codes.
    %
    %   A gbcModel is what GBCTRAIN returns. It can be handed to any of the
    %   functional entry points (GBCPREDICT, GBCSAMPLE, GBCQUANTILE,
    %   GBCMETRICS), and it exposes the methods an outer MCMC sampler needs:
    %
    %     predict         one or more REPRODUCIBLE predictive draws
    %     predictMean     the predictive mean (deterministic)
    %     predictQuantile quantiles at levels you name
    %     sample          fresh i.i.d. predictive draws (exploratory use)
    %     setSamples      re-draw the fixed sample set
    %
    %   CALIBRATION AND THE SAMPLE SET. A GBC surrogate represents its
    %   predictive law implicitly: y = q_hat_tau(x) with tau ~ U[0,1], so a
    %   single uniform tau IS one draw of the surrogate. To let an outer
    %   sampler treat surrogate uncertainty as a latent variable - the way a
    %   GP or BASS emulator's posterior draws are treated - the model carries
    %   a FIXED set of tau draws in obj.samples, drawn once from a private
    %   stream seeded by obj.sampleSeed. Index j of that set names one
    %   realisation of the quantile surface, for the life of the object:
    %
    %       yj = model.predict(X, 'idxSamples', j);    % always the same yj
    %
    %   That reproducibility is the point. If every call drew fresh uniforms,
    %   a Metropolis ratio would compare two different surrogate realisations
    %   and the chain would not target the posterior it was written down for.
    %   It also means an index can be updated in its own Gibbs step, and that
    %   a chain can be replayed exactly from the stored seed.
    %
    %   MULTIPLE TEST POINTS. X may hold any number of rows. A draw is by
    %   default SHARED across those rows - one tau, evaluated at every point -
    %   so a draw is a realisation of the quantile surface as a function, and
    %   a vector-valued prediction keeps its shape rather than acquiring
    %   independent noise at each point. Pass Shared = false for an
    %   independent (still reproducible) level per point, which is what you
    %   want when the rows are unrelated draws rather than one output vector.
    %
    %   ORIENTATION. predict returns numel(idxSamples)-by-size(X,1): draws
    %   down the rows, test points across the columns.
    %
    %   See also GBCTRAIN, GBCPREDICT, GBCSAMPLE, GBCQUANTILE, GBCEVALTAU.

    properties
        params          % learnable parameters (struct of dlarrays)
        opts            % option struct the fit used
        muX             % 1-by-d input centring
        sdX             % 1-by-d input scaling
        muY             % response centring
        sdY             % response scaling
        dIn             % number of predictors
        history         % training history
        numTrain        % training set size
        samples         % 1-by-nSamples fixed tau draws in [0,1]
        sampleSeed      % seed that generated obj.samples
    end

    properties (Dependent)
        nSamples        % numel(obj.samples): the valid range of idxSamples
    end

    methods
        function obj = gbcModel(params, opts, muX, sdX, muY, sdY, d, ...
                                history, numTrain, nv)
            %GBCMODEL Construct a fitted surrogate.
            %
            %   obj = GBCMODEL(params,opts,muX,sdX,muY,sdY,d,history,numTrain)
            %   obj = GBCMODEL(...,'NumSamples',B,'SampleSeed',s) sizes and seeds
            %   the fixed sample set. Either falls back to the matching field
            %   of opts, then to 500 draws seeded from opts.Seed (else 0).
            arguments
                params            = struct()
                opts              = gbcOptions()
                muX               = 0
                sdX               = 1
                muY               = 0
                sdY               = 1
                d                 = []
                history           = []
                numTrain          = []
                nv.NumSamples     = []
                nv.SampleSeed     = []
            end

            obj.params   = params;
            obj.opts     = opts;
            obj.muX      = muX;
            obj.sdX      = sdX;
            obj.muY      = muY;
            obj.sdY      = sdY;
            obj.dIn      = d;
            obj.history  = history;
            obj.numTrain = numTrain;

            % Name-value wins; then opts (so GBCTRAIN can carry the choice
            % through a fit); then the defaults.
            nSamp = firstNonEmpty(nv.NumSamples, optField(opts,'NumSamples'), 500);
            % Inherit the fit's seed so that ensemble members, which differ
            % only in opts.Seed, get different sample sets.
            seed  = firstNonEmpty(nv.SampleSeed, optField(opts,'SampleSeed'), ...
                                  optField(opts,'Seed'), 0);
            obj = obj.setSamples(nSamp, seed);
        end

        function n = get.nSamples(obj)
            n = numel(obj.samples);
        end

        function obj = setSamples(obj, nSamples, seed)
            %SETSAMPLES Re-draw the fixed set of tau draws.
            %
            %   obj = obj.SETSAMPLES(nSamples) redraws from the current seed.
            %   obj = obj.SETSAMPLES(nSamples,seed) sets the seed too.
            %
            %   The draws come from a private RandStream, so this never
            %   touches the global stream - calling it cannot perturb the
            %   reproducibility of anything else in the session.
            arguments
                obj
                nSamples (1,1) double {mustBePositive, mustBeInteger}
                seed = []
            end
            if isempty(seed)
                seed = obj.sampleSeed;          % keep the current seed
            end
            if isempty(seed)
                seed = 0;                       % freshly constructed object
            end
            validateattributes(seed, {'numeric'}, ...
                {'scalar','nonnegative','integer','finite'}, ...
                'setSamples', 'seed');
            obj.sampleSeed = seed;
            obj.samples    = drawTauSet(seed, nSamples);
        end

        function pred = predict(obj, x_new, options)
            %PREDICT Reproducible predictive draws at one or more inputs.
            %
            %   pred = PREDICT(obj,X) returns every stored draw:
            %   obj.nSamples-by-size(X,1).
            %
            %   pred = PREDICT(obj,X,'idxSamples',j) returns draw j alone, as a
            %   1-by-size(X,1) row. This is the MCMC path: the same j always
            %   gives the same numbers, and it costs ONE network evaluation
            %   per test point rather than a whole sample set.
            %
            %   pred = PREDICT(obj,X,'idxSamples',[3 17 42]) returns those three
            %   draws, one per row.
            %
            %   pred = PREDICT(obj,X,'Shared',false) gives each test point its
            %   own independent level within a draw (see the class help). The
            %   draw index then labels a random stream rather than selecting a
            %   stored tau, so the values differ from the Shared = true ones;
            %   they are equally reproducible, but only for a fixed number of
            %   test points.
            %
            %   pred = PREDICT(obj,X,'B',k) is shorthand for idxSamples = 1:k.
            arguments
                obj
                x_new (:,:) double
                options.idxSamples = [];
                options.B          = [];
                options.Shared (1,1) logical = true;
            end

            if isempty(obj.samples)
                error('gbcModel:NoSamples', ...
                    ['This model has no sample set; call ' ...
                     'obj = obj.setSamples(nSamples) first.']);
            end

            idx = options.idxSamples;
            % [] is the documented default. NaN is accepted because it was the
            % old sentinel, so callers written against that keep working.
            if isempty(idx) || (isnumeric(idx) && all(isnan(idx(:))))
                if isempty(options.B)
                    idx = 1:obj.nSamples;
                else
                    validateattributes(options.B, {'numeric'}, ...
                        {'scalar','positive','integer'}, 'predict', 'B');
                    if options.B > obj.nSamples
                        error('gbcModel:TooFewSamples', ...
                            ['Asked for %d draws but the model stores %d; ' ...
                             'call obj = obj.setSamples(%d) first.'], ...
                            options.B, obj.nSamples, options.B);
                    end
                    idx = 1:options.B;
                end
            elseif islogical(idx)
                if numel(idx) ~= obj.nSamples
                    error('gbcModel:BadIndex', ...
                        ['A logical idxSamples must have one element per ' ...
                         'stored draw (%d), got %d.'], obj.nSamples, numel(idx));
                end
                idx = find(idx);
            else
                validateattributes(idx, {'numeric'}, ...
                    {'vector','positive','integer','<=',obj.nSamples}, ...
                    'predict', 'idxSamples');
            end

            idx = reshape(idx, 1, []);

            if options.Shared
                tau = obj.samples(idx);                  % 1-by-k, shared
            else
                tau = obj.tauPerPoint(idx, size(x_new,1));   % n-by-k
            end

            % n-by-k on the response scale; transpose so draws run down the
            % rows, which is the orientation calibration codes expect.
            pred = gbcEvalTau(obj, x_new, tau).';
        end

        function mu = predictMean(obj, x_new, nGrid)
            %PREDICTMEAN Predictive mean, deterministically.
            %
            %   mu = PREDICTMEAN(obj,X) returns a size(X,1)-by-1 vector.
            %
            %   E[Y|x] = integral of q_tau(x) dtau over [0,1], so the mean is
            %   the midpoint-rule average of the quantile curve - no sampling
            %   noise, which is what you want when the surrogate is to be
            %   treated as a fixed mean function rather than sampled.
            arguments
                obj
                x_new (:,:) double
                nGrid (1,1) double {mustBePositive, mustBeInteger} = 256
            end
            grid = ((1:nGrid) - 0.5) / nGrid;
            mu   = mean(gbcEvalTau(obj, x_new, grid), 2);
        end

        function Q = predictQuantile(obj, x_new, probs)
            %PREDICTQUANTILE Predictive quantiles at the levels you name.
            %
            %   Q = PREDICTQUANTILE(obj,X,probs) returns
            %   size(X,1)-by-numel(probs), column j at level probs(j).
            arguments
                obj
                x_new (:,:) double
                probs (1,:) double = [0.05 0.5 0.95]
            end
            Q = gbcQuantile(obj, x_new, probs);
        end

        function S = sample(obj, x_new, B)
            %SAMPLE Fresh i.i.d. predictive draws - NOT reproducible.
            %
            %   S = SAMPLE(obj,X,B) returns size(X,1)-by-B draws with a new
            %   tau per (point,draw) off the global stream. Use it to
            %   summarise the predictive law; use PREDICT inside an MCMC
            %   chain, where a redrawn surrogate would break the chain.
            arguments
                obj
                x_new (:,:) double
                B (1,1) double {mustBePositive, mustBeInteger} = 200
            end
            S = gbcSample(obj, x_new, B);
        end
    end

    methods (Access = private)
        function tau = tauPerPoint(obj, idx, n)
            %TAUPERPOINT Reproducible independent levels, one per test point.
            %   Draw j is the first n values of a private stream labelled by
            %   the seed and j, so it is reproducible for a fixed n but is not
            %   nested across different n.
            tau = zeros(n, numel(idx));
            for j = 1:numel(idx)
                s = RandStream('twister', 'Seed', ...
                               mod(obj.sampleSeed + idx(j), 2^32));
                tau(:,j) = rand(s, n, 1);
            end
        end
    end
end

% =========================================================================

function v = firstNonEmpty(varargin)
%FIRSTNONEMPTY First non-empty argument; [] if every one is empty.
v = [];
for k = 1:numel(varargin)
    if ~isempty(varargin{k})
        v = varargin{k};
        return
    end
end
end

function v = optField(opts, name)
%OPTFIELD opts.(name) when opts is a struct that has it, [] otherwise.
v = [];
if isstruct(opts) && isfield(opts, name)
    v = opts.(name);
end
end

function tau = drawTauSet(seed, nSamples)
%DRAWTAUSET The fixed tau draws, off a private stream.
s   = RandStream('twister', 'Seed', mod(seed, 2^32));
tau = rand(s, 1, nSamples);
end
