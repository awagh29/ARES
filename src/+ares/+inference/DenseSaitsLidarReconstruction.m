classdef DenseSaitsLidarReconstruction < matlab.System
    % Full-resolution stage-one SAITS; original VFH/PID remains in control.
    properties (Nontunable)
        ArtifactPath = ''
    end
    properties (Access=private)
        Model
        Buffer
        Observed
        LastTime = -Inf
        Cached
    end
    methods
        function obj=DenseSaitsLidarReconstruction(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    methods (Access=protected)
        function setupImpl(obj)
            artifact=load(obj.ArtifactPath,"model"); obj.Model=artifact.model;
            assert(string(obj.Model.schemaVersion)=="ares-dense-saits-single-episode-mat-v1", ...
                "ARES:WrongDenseSaitsSchema","Wrong dense SAITS artifact.");
            resetImpl(obj);
        end
        function resetImpl(obj)
            obj.Buffer=zeros(double(obj.Model.nSteps),10752,"single");
            obj.Observed=false(size(obj.Buffer));
            obj.LastTime=-Inf; obj.Cached=cell(1,13);
        end
        function [cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense] = ...
                stepImpl(obj,nominalPoints,position,velocity,eulerZYX,angularRates,enabled,time) %#ok<INUSD>
            time=double(time);
            if time==obj.LastTime
                [cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense]=obj.Cached{:}; return;
            end
            assert(isfinite(time) && time>obj.LastTime,"ARES:DenseSaitsTimeReversed","Invalid scan time.");
            if isfinite(obj.LastTime)
                assert(abs(time-obj.LastTime-.1)<1e-6,"ARES:DenseSaitsScanRate","Expected 10 Hz scans.");
            end
            p=reshape(single(position),1,3); v=reshape(single(velocity),1,3);
            rpy=reshape(single(eulerZYX),1,3); rpy=rpy([3 2 1]);
            context=[p v reshape([sin(rpy);cos(rpy)],1,6)];
            assert(all(isfinite(context)),"ARES:DenseSaitsState","Motion state is nonfinite.");
            available=logical(enabled>=.5); native=true; ready=true;
            raw=zeros(1,10752,"single"); observed=false(1,10752);
            raw(10741:10752)=context; observed(10741:10752)=true;
            cloud=NaN(10740,3,"single");
            if available
                assert(isequal(size(nominalPoints),[10740 3]),"ARES:DenseSaitsCloudShape","Expected organized cloud.");
                points=double(nominalPoints);
                ranges=sqrt(sum(points.^2,2));
                returns=all(isfinite(points),2)&isfinite(ranges)&ranges>0&ranges<=7+7e-6;
                ranges(~returns)=7; ranges(returns)=min(ranges(returns),7);
                raw(1:10740)=single(ranges); observed(1:10740)=true;
                cloud=single(nominalPoints);
            end
            % No oracle access in outage branch; no prediction is fed back as
            % a measured scan. Initial padding is explicitly all-unobserved.
            obj.Buffer(1:end-1,:)=obj.Buffer(2:end,:); obj.Buffer(end,:)=raw;
            obj.Observed(1:end-1,:)=obj.Observed(2:end,:); obj.Observed(end,:)=observed;
            features=raw.'; mask=observed.';
            used=false; valid=false; reason=uint8(0); elapsed=0;
            predicted=zeros(14,1,"single"); sectors=predicted;
            dense=zeros(10740,1,"single");
            if available
                sectors=ares.inference.sectorFeatures(cloud).';
            else
                timer=tic;
                ranges=ares.inference.predictDenseSaits(obj.Model,obj.Buffer,obj.Observed);
                elapsed=toc(timer);
                valid=all(isfinite(ranges),"all");
                if valid
                    dense=ranges(:); cloud=ares.inference.cnnRangeCloud(ranges);
                    predicted=ares.inference.sectorFeatures(cloud).'; sectors=predicted;
                    used=true; reason=uint8(3);
                else
                    reason=uint8(2);
                end
            end
            obj.LastTime=time;
            obj.Cached={cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense};
        end
        function n=getNumInputsImpl(~), n=7; end
        function n=getNumOutputsImpl(~), n=13; end
        function value=getSampleTimeImpl(obj)
            value=createSampleTime(obj,"Type","Discrete","SampleTime",.1);
        end
        function varargout=getOutputSizeImpl(~)
            varargout={[10740 3],[1 1],[1 1],[1 1],[1 1],[1 1],[14 1],[14 1],[10752 1],[10752 1],[1 1],[1 1],[10740 1]};
        end
        function varargout=getOutputDataTypeImpl(~)
            varargout={'single','logical','logical','logical','uint8','double', ...
                'single','single','single','logical','logical','logical','single'};
        end
        function varargout=isOutputComplexImpl(~), varargout=repmat({false},1,13); end
        function varargout=isOutputFixedSizeImpl(~), varargout=repmat({true},1,13); end
    end
end
