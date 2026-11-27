classdef MapCorrectionController < matlab.System
    %MAPCORRECTIONCONTROLLER Map route + live CBF-QP; no learned weights/Python.
    properties (Nontunable)
        PlanFile = ''
    end
    properties (Access=private)
        Options
        Map
        Route
        Index = 2
        Previous = [0 0 0]
        LastTime = -Inf
        LastScanIndex = -1
        MeasuredWorld = zeros(0,3)
        Cached
    end
    methods
        function obj=MapCorrectionController(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    methods (Access=protected)
        function setupImpl(obj)
            data=load(obj.PlanFile,'options','map','plan');
            obj.Options=data.options; obj.Map=data.map; obj.Route=data.plan.WaypointsNedM;
            resetImpl(obj);
        end
        function resetImpl(obj)
            obj.Index=2; obj.Previous=[0 0 0]; obj.LastTime=-Inf; obj.Cached=cell(1,4);
            obj.LastScanIndex=-1; obj.MeasuredWorld=zeros(0,3);
        end
        function [cloud,command,packet,stop]=stepImpl(obj,points,position,velocity,eulerZYX,enabled,time)
            time=double(time);
            if time==obj.LastTime
                [cloud,command,packet,stop]=obj.Cached{:}; return;
            end
            o=obj.Options; timer=tic;
            assert(isfinite(time) && time>=0 && (isinf(obj.LastTime) || ...
                abs(time-obj.LastTime-o.ControlSampleTimeS)<1e-6), ...
                'ARES:MapControlClock','Expected chronological control ticks.');
            p=reshape(double(position),1,3); v=reshape(double(velocity),1,3); euler=reshape(double(eulerZYX),1,3);
            assert(all(isfinite([p v euler])),'ARES:MapNonfiniteState','Motion state must be finite.');
            available=logical(enabled>=.5);
            measuredWorld=zeros(0,3);
            if available
                cloud=single(points);
                % Supplement the static map with actual measured surfaces.
                % Only the nearest valid return per angular bin is retained.
                scanIndex=floor(time/.1+1e-7);
                if scanIndex~=obj.LastScanIndex
                    obj.MeasuredWorld=measuredSurfaces(obj,points,p,euler);
                    obj.LastScanIndex=scanIndex;
                end
                % Preserve the original scan pose while the 20-Hz controller
                % holds a 10-Hz sensor observation between acquisition ticks.
                measuredWorld=obj.MeasuredWorld;
            else
                % Do not inspect, summarize or reuse the unavailable raw cloud.
                cloud=ares.navigation.virtualLidar(obj.Map,p,euler);
            end
            while obj.Index<size(obj.Route,1) && norm(p-obj.Route(obj.Index,:))<=o.WaypointToleranceM
                obj.Index=obj.Index+1;
            end
            reference=obj.Route(obj.Index,:); delta=reference-p; distance=norm(delta);
            speed=min([o.CruiseSpeedMps,o.PositionGain*distance, ...
                sqrt(2*o.BrakingDecelerationMps2*max(distance-.1,0))]);
            desiredVelocity=speed*delta/max(distance,1e-12);
            nominal=o.VelocityGain*(desiredVelocity-v);
            [u,info]=ares.navigation.correctMotion(p,v,nominal,obj.Previous,obj.Map,o,measuredWorld);
            % Exact static NED allocator of the existing guidance plant.
            % Do not clip roll/pitch/thrust after the safety filter.
            yaw=euler(1); vertical=9.81-u(3);
            pitch=atan2(-u(1)*cos(yaw)-u(2)*sin(yaw),vertical);
            roll=atan2((-u(1)*sin(yaw)+u(2)*cos(yaw))*cos(pitch),vertical);
            thrust=.1*vertical/(cos(pitch)*cos(roll));
            heading=atan2(delta(2),delta(1));
            if norm(delta(1:2))<.1, heading=yaw; end
            yawRate=max(-.5,min(.5,atan2(sin(heading-yaw),cos(heading-yaw))));
            command=single([roll;pitch;yawRate;thrust]);
            goal=norm(p-o.GoalState(1:3))<=o.GoalPositionToleranceM && ...
                norm(v-o.GoalState(4:6))<=o.GoalVelocityToleranceMps;
            % Keep a nonempty native trajectory even when start equals goal.
            stop=~info.Feasible || (goal && time>=o.ControlSampleTimeS);
            sectors=double(ares.inference.sectorFeatures(cloud));
            % The viewer shows measured scans only. Keep virtual observations
            % internal, and clear the previous rendered scan during outages.
            if ~available, cloud(:)=NaN; end
            packet=[double(command);u';nominal';reference';p';v'; ...
                double(available);double(~available);double(info.Feasible);info.MaximumViolation; ...
                info.MinimumH;info.MinimumPsi1;double(goal);double(obj.Index);toc(timer);1;sectors'];
            assert(numel(packet)==43 && all(isfinite(packet)) && ...
                all(abs(command(1:2))<=25*pi/180), 'ARES:MapInvalidOutput','Invalid allocated correction output.');
            obj.Previous=u; obj.LastTime=time; obj.Cached={cloud,command,packet,logical(stop)};
        end
        function world=measuredSurfaces(obj,points,p,euler)
            xyz=double(points); ranges=vecnorm(xyz,2,2);
            valid=all(isfinite(xyz),2) & ranges>0 & ranges<=obj.Map.MaxRangeM+1e-5;
            rotation=eul2rotm(euler,'ZYX');
            body=xyz*obj.Map.SensorToBodyRotation'+obj.Map.SensorMountBodyM;
            allWorld=body*rotation'+p;
            % Full boxes already protect mapped surfaces. Adding many point
            % CBFs for those same faces creates needless, changing derivative
            % constraints. Only surfaces outside the map's uncertainty band
            % supplement its geometry (map-residual obstacle detection).
            for obstacle=1:size(obj.Map.BoxMin,1)
                center=(obj.Map.BoxMin(obstacle,:)+obj.Map.BoxMax(obstacle,:))/2;
                half=(obj.Map.BoxMax(obstacle,:)-obj.Map.BoxMin(obstacle,:))/2;
                q=abs(allWorld-center)-half;
                distance=vecnorm(max(q,0),2,2)+min(max(q,[],2),0);
                valid=valid & distance>obj.Options.GeometryUncertaintyM;
            end
            % 15 elevation samples x 716 azimuth samples, elevation fastest.
            [el,az]=ndgrid(1:15,1:716);
            bins=ceil(el(:)/5)+3*floor((az(:)-1)*12/716);
            selected=zeros(36,1); count=0;
            for k=1:36
                ids=find(valid & bins==k);
                if isempty(ids), continue; end
                [~,j]=min(ranges(ids)); count=count+1; selected(count)=ids(j);
            end
            world=allWorld(selected(1:count),:);
        end
        function n=getNumInputsImpl(~), n=6; end
        function n=getNumOutputsImpl(~), n=4; end
        function sample=getSampleTimeImpl(obj)
            % Block itself inherits the explicit sample-time of its inputs.
            sample=createSampleTime(obj,'Type','Inherited');
        end
        function varargout=getOutputSizeImpl(~), varargout={[10740 3],[4 1],[43 1],[1 1]}; end
        function varargout=getOutputDataTypeImpl(~), varargout={'single','single','double','logical'}; end
        function varargout=isOutputComplexImpl(~), varargout={false,false,false,false}; end
        function varargout=isOutputFixedSizeImpl(~), varargout={true,true,true,true}; end
    end
end
