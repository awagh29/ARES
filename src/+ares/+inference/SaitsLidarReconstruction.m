classdef SaitsLidarReconstruction < matlab.System
    % Online, scan-rate SAITS state and source selection for simulation.
    % The nominal outage cloud is never read to construct input features.
    properties (Nontunable)
        ArtifactPath = ''
        UseMex = true
    end
    properties (Access=private)
        Model
        Buffer
        Count = 0
        LastHealthyPosition = zeros(1,3)
        LastHealthyTime = NaN
        LastTime = -Inf
        Cached
        MexFunction = []
        MexIncludesModel = false
    end
    methods
        function obj = SaitsLidarReconstruction(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    methods (Access=protected)
        function setupImpl(obj)
            artifact = load(obj.ArtifactPath,"model","provenance");
            obj.Model = artifact.model;
            assert(obj.Model.nSteps==50 && obj.Model.nFeatures==31 && obj.Model.nTargets==14, ...
                "ARES:InvalidSaitsSchema","Expected the trained 50/31/14 schema.");
            obj.MexFunction = [];
            folder = fileparts(obj.ArtifactPath);
            mexMetadata = fullfile(folder,"mex","saits_endpoint_mex_manifest.json");
            if obj.UseMex && isfile(mexMetadata)
                metadata = jsondecode(fileread(mexMetadata));
                if isfield(metadata,"modelSha256") && metadata.allParityChecksPassed && ...
                        string(metadata.modelSha256)==ares.inference.fileSha256(obj.ArtifactPath) && ...
                        string(metadata.predictorSourceSha256)==ares.inference.fileSha256(which("ares.inference.predictSaits")) && ...
                        string(metadata.mexSourceSha256)==ares.inference.fileSha256(fullfile(fileparts(mfilename("fullpath")),"saits_endpoint_mex.cpp"))
                    mexFile = fullfile(folder,"mex","saits_endpoint_mex."+mexext);
                    if isfile(mexFile) && string(metadata.mexSha256)==ares.inference.fileSha256(mexFile)
                        addpath(fullfile(folder,"mex"));
                        clear saits_endpoint_mex;
                        obj.MexFunction = @saits_endpoint_mex;
                        obj.MexIncludesModel = logical(metadata.mexIncludesModelArgument);
                    end
                end
            end
            resetImpl(obj);
        end
        function resetImpl(obj)
            obj.Buffer = NaN(50,31,"single");
            obj.Count = 0;
            obj.LastHealthyPosition = zeros(1,3);
            obj.LastHealthyTime = NaN;
            obj.LastTime = -Inf;
            obj.Cached = cell(1,12);
        end
        function [cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,backendMex] = ...
                stepImpl(obj,nominalPoints,position,velocity,eulerZYX,angularRates,enabled,time)
            time = double(time);
            % Repeated output evaluations at the same tick must not append
            % duplicate history or change model state.
            if time==obj.LastTime
                [cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,backendMex] = obj.Cached{:};
                return;
            end
            assert(time>obj.LastTime,"ARES:SaitsTimeReversed","Simulation time moved backwards.");
            available = logical(enabled>=.5);
            backendMex = ~isempty(obj.MexFunction);
            p = reshape(double(position),1,3);
            v = reshape(double(velocity),1,3);
            rpy = reshape(double(eulerZYX),1,3);
            rpy = rpy([3 2 1]);
            angular = reshape(double(angularRates),1,3);
            assert(all(isfinite([p v rpy angular time])), ...
                "ARES:InvalidSaitsState","Nonfinite motion state.");
            raw = NaN(1,31,"single");
            if available
                raw(1:14) = ares.inference.sectorFeatures(nominalPoints);
                obj.LastHealthyPosition = p;
                obj.LastHealthyTime = time;
            end
            hasHealthy = isfinite(obj.LastHealthyTime);
            if hasHealthy
                displacement = p-obj.LastHealthyPosition;
                age = time-obj.LastHealthyTime;
            else
                displacement = zeros(1,3);
                age = 0;
            end
            orientation = reshape([sin(rpy);cos(rpy)],1,6);
            raw(15:31) = single([displacement,v,orientation,angular,double(available),age]);
            obj.Buffer(1:49,:) = obj.Buffer(2:50,:);
            obj.Buffer(50,:) = raw;
            obj.Count = min(obj.Count+1,50);
            ready = obj.Count==50 && hasHealthy;
            used = false;
            valid = false;
            reason = uint8(0); % 0 measured;1 warmup;2 invalid;3 model
            elapsed = 0;
            predicted = zeros(14,1,"single");
            sectors = zeros(14,1,"single");
            % Unavailable sensor data is dropped BEFORE inference/controller.
            cloud = NaN(10740,3,"single");
            if available
                cloud = single(nominalPoints);
                sectors = raw(1:14).';
            elseif ~ready
                reason = uint8(1);
            else
                start = tic;
                observed = isfinite(obj.Buffer);
                if isempty(obj.MexFunction)
                    prediction = ares.inference.predictSaits(obj.Model,obj.Buffer,observed);
                else
                    if obj.MexIncludesModel
                        prediction = obj.MexFunction(obj.Model,obj.Buffer,observed);
                    else
                        prediction = obj.MexFunction(obj.Buffer,observed);
                    end
                end
                elapsed = toc(start);
                valid = numel(prediction)==14 && all(isfinite(prediction));
                if valid
                    predicted = single(prediction(:));
                    [cloud,clipped] = ares.inference.sectorProxyCloud(prediction);
                    sectors = clipped(:);
                    used = true;
                    reason = uint8(3);
                else
                    reason = uint8(2);
                end
            end
            features = raw.';
            mask = isfinite(features);
            features(~mask) = 0;
            obj.LastTime = time;
            obj.Cached = {cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,backendMex};
        end
        function count = getNumInputsImpl(~), count = 7; end
        function count = getNumOutputsImpl(~), count = 12; end
        function sample = getSampleTimeImpl(obj)
            sample = createSampleTime(obj,"Type","Discrete","SampleTime",.1);
        end
        function varargout = getOutputSizeImpl(~)
            varargout = {[10740 3],[1 1],[1 1],[1 1],[1 1],[1 1],[14 1],[14 1],[31 1],[31 1],[1 1],[1 1]};
        end
        function varargout = getOutputDataTypeImpl(~)
            varargout = {'single','logical','logical','logical','uint8','double', ...
                'single','single','single','logical','logical','logical'};
        end
        function varargout = isOutputComplexImpl(~)
            varargout = repmat({false},1,12);
        end
        function varargout = isOutputFixedSizeImpl(~)
            varargout = repmat({true},1,12);
        end
    end
end
