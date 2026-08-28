classdef gbcModel
    %GBCMODEL GBC NN model definition

    properties
        params
        opts
        muX
        sdX
        muY
        sdY
        dIn
        history
        numTrain
    end

    methods
        function obj = gbcModel(params, opts, muX, sdX, muY, sdY, d, history, numTrain)
            obj.params = params;
            obj.opts = opts;
            obj.muX = muX;
            obj.sdX = sdX;
            obj.muY = muY;
            obj.sdY = sdY;
            obj.dIn = d;
            obj.history = history;
            obj.numTrain = numTrain;
        end

        function pred = predict(obj, x_new, options)
            arguments
                obj
                x_new
                options.idxSamples = nan;
                options.B = 500;
            end
            idxSamples = options.idxSamples;
            if isnan(idxSamples) 
                idxSamples = 1:options.B;
            end

            pred = gbcSample(obj, x_new, options.B);
            pred = pred(:, idxSamples);

        end
    end
end