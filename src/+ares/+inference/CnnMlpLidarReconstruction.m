classdef CnnMlpLidarReconstruction < matlab.System
    % State-only dense CNN replacement, selected only during sensor outages.
    properties (Nontunable)
        ArtifactPath = ''
    end
    properties (Access=private)
        Model
        LastTime = -Inf
        Cached
    end
    methods
        function obj=CnnMlpLidarReconstruction(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    methods (Access=protected)
        function setupImpl(obj)
            artifact=load(obj.ArtifactPath,"model");
            assert(string(artifact.model.schemaVersion)=="ares-cnn-mlp-single-episode-mat-v1", ...
                "ARES:InvalidCnnSchema","Wrong CNN artifact.");
            obj.Model=artifact.model;
            resetImpl(obj);
        end
        function resetImpl(obj)
            obj.LastTime=-Inf; obj.Cached=cell(1,13);
        end
        function [cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense] = ...
                stepImpl(obj,nominalPoints,position,velocity,eulerZYX,angularRates,enabled,time) %#ok<INUSD>
            time=double(time);
            if time==obj.LastTime
                [cloud,used,ready,valid,reason,elapsed,predicted,sectors,features,mask,available,native,dense]=obj.Cached{:};
                return;
            end
            assert(isfinite(time) && time>obj.LastTime,"ARES:CnnTimeReversed","Invalid scan time.");
            p=reshape(double(position),1,3); v=reshape(double(velocity),1,3);
            rpy=reshape(double(eulerZYX),1,3); rpy=rpy([3 2 1]);
            raw=single([p,v,rpy]);
            assert(all(isfinite(raw)),"ARES:InvalidCnnState","Nonfinite state.");
            angles=raw(7:9);
            features=[raw(1:6),reshape([sin(angles);cos(angles)],1,6)].';
            mask=true(12,1); available=logical(enabled>=.5); native=true;
            ready=true; used=false; valid=false; reason=uint8(0); elapsed=0;
            predicted=zeros(14,1,"single"); sectors=zeros(14,1,"single");
            dense=zeros(10740,1,"single"); % Unused unless reconstruction_used is true.
            cloud=NaN(10740,3,"single");
            if available
                cloud=single(nominalPoints);
                sectors=ares.inference.sectorFeatures(cloud).';
            else
                % nominalPoints is deliberately never inspected in this branch.
                timer=tic;
                ranges=ares.inference.predictCnnMlp(obj.Model,raw);
                elapsed=toc(timer);
                valid=all(isfinite(ranges),"all");
                if valid
                    dense=ranges(:);
                    cloud=ares.inference.cnnRangeCloud(ranges);
                    predicted=ares.inference.sectorFeatures(cloud).';
                    sectors=predicted; used=true; reason=uint8(3);
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
            varargout={[10740 3],[1 1],[1 1],[1 1],[1 1],[1 1],[14 1],[14 1],[12 1],[12 1],[1 1],[1 1],[10740 1]};
        end
        function varargout=getOutputDataTypeImpl(~)
            varargout={'single','logical','logical','logical','uint8','double', ...
                'single','single','single','logical','logical','logical','single'};
        end
        function varargout=isOutputComplexImpl(~), varargout=repmat({false},1,13); end
        function varargout=isOutputFixedSizeImpl(~), varargout=repmat({true},1,13); end
    end
end
