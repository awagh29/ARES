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
        Options
    end
    methods
        function obj=GoalHoldReference(varargin), setProperties(obj,nargin,varargin{:}); end
    end
    methods (Access=protected)
        function setupImpl(obj)
            mission=load(obj.PlanFile,'options'); o=mission.options;
            obj.Goal=o.GoalState(1:3)';
            obj.Options=ares.mapTemporal.validateOptions(o);
            if obj.Options.EnableReferenceGovernor
                obj.Radius=obj.Options.GoalApproachRadiusM;
            else
                obj.Radius=min(.75,o.MinimumSurfaceClearanceM/2);
            end
        end
        function resetImpl(obj), obj.Holding=false; obj.HoldYaw=0; end
        function [reference,yaw,holding]=stepImpl(obj,nominalReference,nominalYaw,position,eulerZYX,points)
            delta=obj.Goal-double(position(:)); distance=norm(delta);
            clear=true;
            if obj.Options.EnableReferenceGovernor && distance<=obj.Radius
                offsets=ares.mapTemporal.cloudOffsets(points,eulerZYX);
                radius=obj.Options.VehicleRadiusM+obj.Options.MinimumSurfaceClearanceM+obj.Options.GovernorMarginM;
                clear=ares.mapTemporal.corridorDistance(offsets,delta,radius,7)>=distance;
                % Rejoin VFH if a newly selected scan blocks the approach.
                if ~clear, obj.Holding=false; end
            end
            if ~obj.Holding && distance<=obj.Radius && clear
                obj.Holding=true; obj.HoldYaw=double(eulerZYX(1));
            end
            reference=reshape(nominalReference,3,1); yaw=nominalYaw;
            if obj.Holding
                reference=cast(obj.Goal,'like',nominalReference);
                yaw=cast(obj.HoldYaw,'like',nominalYaw);
            end
            holding=obj.Holding;
        end
        function n=getNumInputsImpl(~), n=5; end
        function n=getNumOutputsImpl(~), n=3; end
        function sample=getSampleTimeImpl(obj)
            sample=createSampleTime(obj,'Type','Discrete','SampleTime',.1);
        end
        function varargout=getOutputSizeImpl(~), varargout={[3 1],[1 1],[1 1]}; end
        function varargout=getOutputDataTypeImpl(obj)
            varargout={propagatedInputDataType(obj,1),propagatedInputDataType(obj,2),'logical'};
        end
        function varargout=isOutputComplexImpl(~), varargout={false,false,false}; end
        function varargout=isOutputFixedSizeImpl(~), varargout={true,true,true}; end
    end
end
