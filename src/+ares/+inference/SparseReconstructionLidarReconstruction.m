classdef SparseReconstructionLidarReconstruction < matlab.System
    % Per-scan sparse CNN/FiLM; original VFH/PID remains in control.
    properties (Nontunable)
        ArtifactPath = ''
    end
    properties (Access=private)
        Model
        LastTime = -Inf
        Cached
    end
    methods
        function obj=SparseReconstructionLidarReconstruction(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    methods (Access=protected)
        function setupImpl(obj)
            artifact=load(obj.ArtifactPath,"model"); obj.Model=artifact.model;
            assert(string(obj.Model.schemaVersion)=="ares-sparse-reconstruction-single-episode-mat-v1", ...
                "ARES:WrongSparseSchema","Wrong sparse reconstruction artifact.");
            resetImpl(obj);
        end
        function resetImpl(obj)
            obj.LastTime=-Inf; obj.Cached=cell(1,14);
        end
        function [cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense,hitMask] = ...
                stepImpl(obj,nominalPoints,position,velocity,eulerZYX,angularRates,enabled,time) %#ok<INUSD>
            time=double(time);
            if time==obj.LastTime
                [cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense,hitMask]=obj.Cached{:}; return;
            end
            assert(isfinite(time) && time>obj.LastTime,"ARES:SparseTimeReversed","Invalid scan time.");
            if isfinite(obj.LastTime)
                assert(abs(time-obj.LastTime-.1)<1e-6,"ARES:SparseScanRate","Expected 10 Hz scans.");
            end
            p=reshape(single(position),1,3); v=reshape(single(velocity),1,3);
            rpy=reshape(single(eulerZYX),1,3); rpy=rpy([3 2 1]);
            rawState=[p v rpy];
            assert(all(isfinite(rawState)),"ARES:SparseState","Motion state is nonfinite.");
            context=[p v reshape([sin(rpy);cos(rpy)],1,6)];
            contextMask=true(1,12);
            if ~obj.Model.usePosition, context(1:3)=0; contextMask(1:3)=false; end
            available=logical(enabled>=.5); native=true; ready=true;
            ranges=zeros(15,716,"single"); hits=false(15,716); support=false(15,716);
            cloud=NaN(10740,3,"single");
            % No nominalPoints access in the outage branch. No history, seed
            % scans or previous predictions are injected into this per-scan model.
            if available
                assert(isequal(size(nominalPoints),[10740 3]),"ARES:SparseCloudShape","Expected organized cloud.");
                points=double(nominalPoints);
                measured=sqrt(sum(points.^2,2));
                returns=all(isfinite(points),2)&isfinite(measured)&measured>0&measured<=7+7e-6;
                measured(~returns)=7; measured(returns)=min(measured(returns),7);
                ranges=reshape(single(measured),15,716); hits=reshape(returns,15,716);
                support(:)=true; cloud=single(nominalPoints);
            end
            features=[ranges(:);context.']; mask=[support(:);contextMask.']; hitMask=hits(:);
            used=false; valid=false; reason=uint8(0); elapsed=0;
            predicted=zeros(14,1,"single"); sectors=predicted; dense=zeros(10740,1,"single");
            if available
                sectors=ares.inference.sectorFeatures(cloud).';
            else
                timer=tic;
                reconstruction=ares.inference.predictSparseReconstruction(obj.Model,ranges,hits,support,rawState);
                elapsed=toc(timer);
                valid=all(isfinite(reconstruction),"all");
                if valid
                    dense=reconstruction(:); cloud=ares.inference.cnnRangeCloud(reconstruction);
                    predicted=ares.inference.sectorFeatures(cloud).'; sectors=predicted;
                    used=true; reason=uint8(3);
                else
                    reason=uint8(2);
                end
            end
            obj.LastTime=time;
            obj.Cached={cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense,hitMask};
        end
        function n=getNumInputsImpl(~), n=7; end
        function n=getNumOutputsImpl(~), n=14; end
        function value=getSampleTimeImpl(obj)
            value=createSampleTime(obj,"Type","Discrete","SampleTime",.1);
        end
        function varargout=getOutputSizeImpl(~)
            varargout={[10740 3],[1 1],[1 1],[1 1],[1 1],[1 1],[14 1],[14 1],[10752 1],[10752 1],[1 1],[1 1],[10740 1],[10740 1]};
        end
        function varargout=getOutputDataTypeImpl(~)
            varargout={'single','logical','logical','logical','uint8','double', ...
                'single','single','single','logical','logical','logical','single','logical'};
        end
        function varargout=isOutputComplexImpl(~), varargout=repmat({false},1,14); end
        function varargout=isOutputFixedSizeImpl(~), varargout=repmat({true},1,14); end
    end
end
