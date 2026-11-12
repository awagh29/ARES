classdef SaitsBarrierController < matlab.System
    % Persistent, scan-rate augmented reconstruction + live hard safety QP.
    % Commands directly replace the old VFH/PID output in an experiment copy.
    properties (Nontunable)
        ProjectRoot = ''
        DeploymentPath = ''
        PythonExecutable = ''
        GoalNed = [30 35 -7]
        ContinueAfterFailure = false
    end
    properties (Access=private)
        Runtime
        LastTime = -Inf
        Cached
    end
    methods
        function obj = SaitsBarrierController(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    methods (Access=protected)
        function setupImpl(obj)
            module = ares.inference.loadSaitsBarrierRuntime(string(obj.ProjectRoot),string(obj.PythonExecutable));
            obj.Runtime = module.Controller(char(obj.DeploymentPath),double(obj.GoalNed), ...
                pyargs("continue_after_failure",logical(obj.ContinueAfterFailure)));
            obj.LastTime = -Inf;
            obj.Cached = cell(1,4);
        end
        function resetImpl(obj)
            if ~isempty(obj.Runtime), obj.Runtime.reset(); end
            obj.LastTime = -Inf;
            obj.Cached = cell(1,4);
        end
        function [cloud,command,packed,failed] = stepImpl(obj,points,position,velocity,eulerZYX,omega,enabled,time)
            time = double(time);
            if time==obj.LastTime
                [cloud,command,packed,failed] = obj.Cached{:}; return;
            end
            available = logical(enabled>=.5);
            measured = zeros(1,14);
            if available
                measured = double(ares.inference.sectorFeatures(points));
            end
            % Oracle cloud is discarded before crossing into Python on outage.
            rpy = reshape(double(eulerZYX),1,3); rpy = rpy([3 2 1]);
            result = obj.Runtime.step(reshape(double(position),1,3),reshape(double(velocity),1,3), ...
                rpy,reshape(double(omega),1,3),measured,available,time);
            packed = reshape(double(result),113,1);
            assert(all(isfinite(packed)),"ARES:NonfiniteBarrierOutput","Controller returned invalid output.");
            command = single(packed(1:4));
            failed = logical(packed(110)) && ~obj.ContinueAfterFailure;
            if available
                cloud = single(points);
            else
                cloud = ares.inference.sectorProxyCloud(packed(19:32));
            end
            obj.LastTime = time;
            obj.Cached = {cloud,command,packed,failed};
        end
        function n = getNumInputsImpl(~), n = 7; end
        function n = getNumOutputsImpl(~), n = 4; end
        function sample = getSampleTimeImpl(obj)
            sample = createSampleTime(obj,"Type","Discrete","SampleTime",.1);
        end
        function varargout = getOutputSizeImpl(~)
            varargout = {[10740 3],[4 1],[113 1],[1 1]};
        end
        function varargout = getOutputDataTypeImpl(~)
            varargout = {'single','single','double','logical'};
        end
        function varargout = isOutputComplexImpl(~), varargout = {false,false,false,false}; end
        function varargout = isOutputFixedSizeImpl(~), varargout = {true,true,true,true}; end
    end
end
