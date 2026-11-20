classdef NominalRanges < matlab.System
    % Offline scoring branch only. Never connected to inference or guidance.
    methods (Access=protected)
        function ranges=stepImpl(~,points)
            d=sqrt(sum(double(points).^2,2));
            hits=all(isfinite(points),2)&isfinite(d)&d>0&d<=7+7e-6;
            d(~hits)=7; d(hits)=min(d(hits),7); ranges=single(d);
        end
        function n=getNumInputsImpl(~), n=1; end
        function n=getNumOutputsImpl(~), n=1; end
        function s=getOutputSizeImpl(~), s=[10740 1]; end
        function s=getOutputDataTypeImpl(~), s='single'; end
        function s=isOutputComplexImpl(~), s=false; end
        function s=isOutputFixedSizeImpl(~), s=true; end
        function s=getSampleTimeImpl(obj), s=createSampleTime(obj,'Type','Discrete','SampleTime',.1); end
    end
end
