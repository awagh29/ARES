classdef GoalHoldReference < matlab.System
    % Hold the requested terminal position using the existing position PID.
    properties (Nontunable)
        PlanFile = ''
    end
    properties (Access=private)
        Goal
        Radius
        Holding = false
        HoldYaw = 0
    end
    methods
        function obj=GoalHoldReference(varargin), setProperties(obj,nargin,varargin{:}); end
    end
    methods (Access=protected)
        function setupImpl(obj)
            mission=load(obj.PlanFile,'options'); o=mission.options;
            obj.Goal=o.GoalState(1:3)';
            obj.Radius=min(.75,o.MinimumSurfaceClearanceM/2);
        end
        function resetImpl(obj), obj.Holding=false; obj.HoldYaw=0; end
        function [reference,yaw,holding]=stepImpl(obj,nominalReference,nominalYaw,position,eulerZYX)
            if ~obj.Holding && norm(double(position(:))-obj.Goal)<=obj.Radius
                obj.Holding=true; obj.HoldYaw=double(eulerZYX(1));
            end
            reference=reshape(nominalReference,3,1); yaw=nominalYaw;
            if obj.Holding
                reference=cast(obj.Goal,'like',nominalReference);
                yaw=cast(obj.HoldYaw,'like',nominalYaw);
            end
            holding=obj.Holding;
        end
        function n=getNumInputsImpl(~), n=4; end
        function n=getNumOutputsImpl(~), n=3; end
        function varargout=getOutputSizeImpl(~), varargout={[3 1],[1 1],[1 1]}; end
        function varargout=getOutputDataTypeImpl(obj)
            varargout={propagatedInputDataType(obj,1),propagatedInputDataType(obj,2),'logical'};
        end
        function varargout=isOutputComplexImpl(~), varargout={false,false,false}; end
        function varargout=isOutputFixedSizeImpl(~), varargout={true,true,true}; end
    end
end
