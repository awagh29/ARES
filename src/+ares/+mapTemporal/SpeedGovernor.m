classdef SpeedGovernor < matlab.System
    % Limit the position-PID velocity request using selected scan and momentum.
    % No CBF, map oracle, network modification, or low-level gain changes.
    properties (Nontunable)
        PlanFile = ''
    end
    properties (Access=private)
        Options
        Previous = [0;0;0]
        Planner
    end
    methods
        function obj=SpeedGovernor(varargin), setProperties(obj,nargin,varargin{:}); end
    end
    methods (Access=protected)
        function setupImpl(obj)
            data=load(obj.PlanFile,'options'); obj.Options=ares.mapTemporal.validateOptions(data.options);
            obj.Previous=reshape(obj.Options.StartState(4:6),3,1);
            if obj.Options.EnableLocalPlanner, obj.Planner=ares.mapTemporal.TrajectoryPlanner(obj.Options); end
        end
        function resetImpl(obj)
            obj.Previous=reshape(obj.Options.StartState(4:6),3,1);
            if ~isempty(obj.Planner), obj.Planner.reset(); end
        end
        function [command,packet,plannerPacket]=stepImpl(obj,request,position,velocity,eulerZYX,points,approaching)
            o=obj.Options; p=double(position(:)); v=double(velocity(:)); u=double(request(:));
            assert(all(isfinite([p;v])),'ARES:TemporalGovernorState','Nonfinite governor state.');
            if ~all(isfinite(u)), u=zeros(3,1); end % invalid VFH direction requests braking
            offsets=ares.mapTemporal.cloudOffsets(points,eulerZYX);
            plannerPacket=zeros(12,1);
            if ~isempty(obj.Planner), [u,plannerPacket]=obj.Planner.step(p,v,offsets); end
            radius=o.VehicleRadiusM+o.MinimumSurfaceClearanceM+o.GovernorMarginM;
            horizon=max(0,7-norm([0 0 -.4])-radius);
            requestedDistance=ares.mapTemporal.corridorDistance(offsets,u,radius,horizon);
            motionDistance=ares.mapTemporal.corridorDistance(offsets,v,radius,horizon);
            a=o.BrakingDecelerationMps2; tau=o.GovernorLatencyS;
            requestedCap=sqrt((a*tau)^2+2*a*requestedDistance)-a*tau;
            motionCap=sqrt((a*tau)^2+2*a*motionDistance)-a*tau;
            cap=min(o.CruiseSpeedMps,requestedCap);
            goalDistance=norm(p-o.GoalState(1:3)');
            if approaching
                cap=min([cap,o.GoalApproachGain*goalDistance, ...
                    sqrt((a*tau)^2+2*a*goalDistance)-a*tau]);
            end
            braking=norm(v)>motionCap+.05;
            if braking, cap=0; end
            target=u*min(1,cap/max(norm(u),1e-12));
            % Bound nominal acceleration; safety reductions bypass smoothing.
            change=target-obj.Previous;
            commandD=obj.Previous+change*min(1,o.GovernorAccelerationMps2*.1/max(norm(change),1e-12));
            commandD=commandD*min(1,cap/max(norm(commandD),1e-12));
            % Direction interpolation can cross a blocked corridor: recheck it.
            blendedDistance=ares.mapTemporal.corridorDistance(offsets,commandD,radius,horizon);
            blendedCap=sqrt((a*tau)^2+2*a*blendedDistance)-a*tau;
            commandD=commandD*min(1,blendedCap/max(norm(commandD),1e-12));
            obj.Previous=commandD; command=cast(commandD,'like',request);
            packet=[u;commandD;cap;requestedDistance;motionDistance;double(braking);double(approaching);goalDistance];
        end
        function n=getNumInputsImpl(~), n=6; end
        function n=getNumOutputsImpl(~), n=3; end
        function sample=getSampleTimeImpl(obj)
            sample=createSampleTime(obj,'Type','Discrete','SampleTime',.1);
        end
        function varargout=getOutputSizeImpl(~), varargout={[3 1],[12 1],[12 1]}; end
        function varargout=getOutputDataTypeImpl(obj), varargout={propagatedInputDataType(obj,1),'double','double'}; end
        function varargout=isOutputComplexImpl(~), varargout={false,false,false}; end
        function varargout=isOutputFixedSizeImpl(~), varargout={true,true,true}; end
    end
end
