classdef LidarReconstruction < matlab.System
    % Known-map CNN/GRU reconstruction at 10 Hz, ahead of original VFH/PID.
    properties (Nontunable)
        ArtifactPath = ''
        PlanFile = ''
    end
    properties (Access=private)
        Model
        Options
        History
        LastTime = -Inf
        Cached
        ScanCount = 0
    end
    methods
        function obj=LidarReconstruction(varargin), setProperties(obj,nargin,varargin{:}); end
    end
    methods (Access=protected)
        function setupImpl(obj)
            artifact=load(obj.ArtifactPath,'model'); obj.Model=artifact.model;
            mission=load(obj.PlanFile,'options'); obj.Options=mission.options;
            assert(string(obj.Model.schemaVersion)=="ares-map-temporal-native-v1", ...
                'ARES:TemporalSchema','Choose a map-temporal checkpoint.');
            resetImpl(obj);
        end
        function resetImpl(obj)
            obj.LastTime=-Inf; obj.Cached=cell(1,16); obj.ScanCount=0;
            obj.History=ares.mapTemporal.emptyHistory(obj.Model);
        end
        function [cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense,hitMask,terminal,packet] = ...
                stepImpl(obj,nominalPoints,position,velocity,eulerZYX,angularRates,enabled,time) %#ok<INUSD>
            time=double(time);
            if time==obj.LastTime
                [cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense,hitMask,terminal,packet]=obj.Cached{:};
                return;
            end
            assert(isfinite(time) && time>obj.LastTime,'ARES:TemporalTime','Scan times must increase.');
            if isfinite(obj.LastTime)
                assert(abs(time-obj.LastTime-.1)<1e-6,'ARES:TemporalScanRate','Expected 10 Hz scans.');
            end
            p=reshape(single(position),1,3); v=reshape(single(velocity),1,3);
            rpy=reshape(single(eulerZYX),1,3); rpy=rpy([3 2 1]); state=[p v rpy];
            assert(all(isfinite(state)),'ARES:TemporalState','Motion state is nonfinite.');
            goal=single(obj.Options.GoalState(1:3));
            available=logical(enabled>=.5); native=true; ready=true;
            ranges=zeros(15,716,'single'); hits=false(15,716); support=false(15,716);
            % The hidden nominal cloud is inspected ONLY when LiDAR is available.
            % Predictions are never inserted into the measured history buffer.
            if available
                assert(isequal(size(nominalPoints),[10740 3]),'ARES:TemporalCloud','Expected organized cloud.');
                points=double(nominalPoints); measured=sqrt(sum(points.^2,2));
                returns=all(isfinite(points),2)&isfinite(measured)&measured>0&measured<=7+7e-6;
                measured(~returns)=7; measured(returns)=min(measured(returns),7);
                ranges=reshape(single(measured),15,716); hits=reshape(returns,15,716);
                support(:)=true; cloud=single(nominalPoints);
            end
            obj.History=ares.mapTemporal.pushHistory(obj.Model,obj.History,ranges,hits,support,state,goal,time);
            context=[p v reshape([sin(rpy);cos(rpy)],1,6)];
            features=[ranges(:);context']; mask=[support(:);true(12,1)]; hitMask=hits(:);
            used=~available; valid=true; reason=uint8(0); elapsed=0;
            % -1 means the map prior was not evaluated on this healthy tick.
            predicted=zeros(14,1,'single'); priorMin=-1; residualMax=0;
            if used
                timer=tic;
                [ranges,diagnostics]=ares.mapTemporal.predict(obj.Model,ranges,hits,support,obj.History);
                elapsed=toc(timer);
                assert(all(isfinite(ranges),'all'),'ARES:TemporalPrediction','Nonfinite learned prediction; stopping simulation.');
                dense=ranges(:); cloud=obj.Model.raysSensor.*dense;
                cloud(dense<=0|dense>=obj.Model.maxRange,:)=NaN;
                priorMin=double(min(diagnostics.Prior,[],'all'));
                residualMax=double(max(abs(diagnostics.Correction),[],'all')); reason=uint8(3);
                predicted=ares.inference.sectorFeatures(cloud)';
            else
                dense=ranges(:);
            end
            sectors=ares.inference.sectorFeatures(cloud)';
            positionError=norm(double(p)-double(goal));
            velocityError=norm(double(v)-obj.Options.GoalState(4:6));
            terminal=time>=.1 && positionError<=obj.Options.GoalPositionToleranceM && ...
                velocityError<=obj.Options.GoalVelocityToleranceMps;
            obj.ScanCount=obj.ScanCount+1;
            packet=[double([p v rpy goal])';double(available);double(used);nnz(obj.History.Present); ...
                double(obj.History.Ages(end));elapsed;positionError;velocityError;double(terminal); ...
                priorMin;residualMax;obj.ScanCount;double(nnz(support))/numel(support)];
            obj.LastTime=time;
            obj.Cached={cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense,hitMask,terminal,packet};
        end
        function n=getNumInputsImpl(~), n=7; end
        function n=getNumOutputsImpl(~), n=16; end
        function value=getSampleTimeImpl(obj), value=createSampleTime(obj,'Type','Discrete','SampleTime',.1); end
        function varargout=getOutputSizeImpl(~)
            varargout={[10740 3],[1 1],[1 1],[1 1],[1 1],[1 1],[14 1],[14 1],[10752 1],[10752 1],[1 1],[1 1],[10740 1],[10740 1],[1 1],[24 1]};
        end
        function varargout=getOutputDataTypeImpl(~)
            varargout={'single','logical','logical','logical','uint8','double', ...
                'single','single','single','logical','logical','logical','single','logical','logical','double'};
        end
        function varargout=isOutputComplexImpl(~), varargout=repmat({false},1,16); end
        function varargout=isOutputFixedSizeImpl(~), varargout=repmat({true},1,16); end
    end
end
