classdef TrajectoryPlanner < handle
    % Selected-scan, receding-horizon velocity sampling with persistent escape.
    % No obstacle-map input, scoring scan, CBF, QP or learned weight updates.
    properties (Access=private)
        Options
        Trail = zeros(0,3)
        Errors = zeros(0,1)
        Previous = [0 0 0]
        RecoveryTarget = [0 0 0]
        RecoveryLeft = 0
        Entries = 0
        BlockedTicks = 0
        Guide
    end
    methods
        function obj=TrajectoryPlanner(options)
            obj.Options=options; obj.Guide=ares.mapTemporal.ObservedRouteGuide(options); obj.reset();
        end
        function reset(obj)
            obj.Trail=zeros(0,3); obj.Errors=zeros(0,1);
            obj.Previous=obj.Options.StartState(4:6);
            obj.RecoveryTarget=[0 0 0]; obj.RecoveryLeft=0;
            obj.Entries=0; obj.BlockedTicks=0;
            obj.Guide.reset();
        end
        function [command,packet]=step(obj,position,velocity,offsets)
            timer=tic; o=obj.Options; p=reshape(double(position),1,3); v=reshape(double(velocity),1,3);
            goal=o.GoalState(1:3); goalError=norm(goal-p);
            assert(all(isfinite([p v offsets(:)'])),'ARES:PlannerNonfinite','Planner state/cloud must be finite.');
            obj.Trail=[obj.Trail;p]; obj.Trail=obj.Trail(max(1,end-149):end,:);
            window=round(o.PlannerStallTimeS/.1);
            % Deliberate escape motion is not a failure to approach the goal.
            % On exit, require a fresh normal-planning progress window before
            % another stall can trigger; otherwise recoveries chain forever.
            if obj.RecoveryLeft>0
                obj.Errors=goalError;
            else
                obj.Errors=[obj.Errors;goalError];
                obj.Errors=obj.Errors(max(1,end-4*window):end);
            end
            % A voxel represents its full cell, never just a discarded centroid.
            voxel=o.PlannerVoxelM;
            centers=unique(floor(offsets/voxel),'rows')*voxel+voxel/2;
            cellPadding=sqrt(3)*voxel/2;
            radius=o.VehicleRadiusM+o.MinimumSurfaceClearanceM+o.GovernorMarginM+cellPadding;
            initial=Inf; if ~isempty(centers), initial=min(vecnorm(centers,2,2)); end
            % If already inside the engineering buffer, allow only trajectories
            % that do not deepen that overlap. Required clearance is not relaxed.
            radius=max(o.VehicleRadiusM+o.MinimumSurfaceClearanceM+cellPadding, ...
                min(radius,initial-.015));
            target=goal;
            if obj.RecoveryLeft>0
                obj.RecoveryLeft=obj.RecoveryLeft-1;
                if norm(obj.RecoveryTarget-p)<.6 || obj.RecoveryLeft==0
                    obj.RecoveryLeft=0; obj.Errors=goalError;
                else
                    target=obj.RecoveryTarget;
                end
            end
            progress=0;
            if numel(obj.Errors)>1, progress=obj.Errors(max(1,end-window))-goalError; end
            displacement=norm(p-obj.Trail(max(1,end-window),:));
            stationary=numel(obj.Errors)>window && progress<o.PlannerProgressM && displacement<.4;
            looping=numel(obj.Errors)>4*window && obj.Errors(1)-goalError<o.PlannerProgressM;
            stalled=(stationary || looping) && goalError>1;
            delta=target-p; direct=delta/max(norm(delta),1e-12);
            az=(0:23)'*2*pi/24;
            vertical=max(-.35,min(.35,direct(3)));
            directions=[cos(az)*sqrt(1-vertical^2),sin(az)*sqrt(1-vertical^2),vertical*ones(24,1)];
            directions=[directions;direct;obj.Previous/max(norm(obj.Previous),1e-12)];
            speed=min(o.CruiseSpeedMps,o.GoalApproachGain*goalError);
            candidates=[directions*speed;directions*(speed*.5);zeros(1,3)];
            [~,endpoints,clearance,feasible]=rollouts(obj,p,v,candidates,centers,radius);
            routeCosts=obj.Guide.step(p,offsets,[p;p+endpoints]);
            endpointCosts=routeCosts(2:end); guideValid=isfinite(endpointCosts) & feasible;
            travel=vecnorm(endpoints,2,2);
            if (stalled || obj.BlockedTicks>=10) && obj.RecoveryLeft<=0
                recoveryScore=travel+.15*(goalError-vecnorm(p+endpoints-goal,2,2));
                recoveryScore=recoveryScore-3*revisit(obj,p+endpoints);
                if any(guideValid)
                    recoveryScore(guideValid)=recoveryScore(guideValid)-endpointCosts(guideValid);
                    recoveryScore(~guideValid)=-Inf;
                end
                recoveryScore(~feasible | travel<.6)=-Inf;
                [best,index]=max(recoveryScore);
                if isfinite(best)
                    direction=endpoints(index,:)/travel(index);
                    free=ares.mapTemporal.corridorDistance(offsets,direction, ...
                        o.VehicleRadiusM+o.MinimumSurfaceClearanceM+o.GovernorMarginM,5);
                    obj.RecoveryTarget=p+direction*min(3.5,.8*free);
                    obj.RecoveryLeft=round(o.PlannerRecoveryTimeS/.1);
                    obj.Entries=obj.Entries+1; obj.Errors=goalError;
                    target=obj.RecoveryTarget;
                end
            end
            localProgress=norm(target-p)-vecnorm(p+endpoints-target,2,2);
            globalProgress=goalError-vecnorm(p+endpoints-goal,2,2);
            heading=candidates*obj.Previous'/max(norm(obj.Previous)*o.CruiseSpeedMps,1e-9);
            preference=min(clearance-radius,o.GovernorPlanningBufferM);
            score=3*localProgress+.15*globalProgress+.35*preference+.25*heading;
            if obj.RecoveryLeft<=0 && any(guideValid) && goalError>2
                anchor=min(endpointCosts(guideValid));
                if isfinite(routeCosts(1)), anchor=routeCosts(1); end
                score=3*(anchor-endpointCosts)+.3*globalProgress+.35*preference+.25*heading;
                score(~guideValid)=-Inf;
            end
            score=score-1.5*revisit(obj,p+endpoints);
            score(~feasible)=-Inf;
            [best,index]=max(score);
            if isfinite(best)
                command=candidates(index,:);
                obj.BlockedTicks=0;
            else
                command=[0 0 0]; index=size(candidates,1);
                obj.BlockedTicks=obj.BlockedTicks+1;
                if obj.BlockedTicks>=10, obj.RecoveryLeft=0; end
            end
            obj.Previous=command;
            packet=[double(obj.RecoveryLeft>0);double(stalled);obj.Entries;sum(feasible); ...
                norm(command);min(7,clearance(index))-cellPadding-o.VehicleRadiusM; ...
                progress;target';double(~isfinite(best));toc(timer)];
            command=command';
        end
    end
    methods (Access=private)
        function penalty=revisit(obj,positions)
            % Ignore the immediately preceding motion; penalize old loops.
            trail=obj.Trail(1:max(0,end-20),:);
            penalty=zeros(size(positions,1),1);
            if isempty(trail), return; end
            d2=max(sum(positions.^2,2)+sum(trail.^2,2)'-2*positions*trail',0);
            penalty=exp(-min(d2,[],2)/2);
        end
        function [paths,endpoints,clearance,feasible]=rollouts(obj,p,v,commands,points,radius)
            o=obj.Options; dt=o.PlannerRolloutStepS; n=size(commands,1);
            steps=ceil(o.PlannerHorizonS/dt);
            brakeSteps=ceil((o.GovernorLatencyS+o.CruiseSpeedMps/o.BrakingDecelerationMps2)/dt)+2;
            total=steps+brakeSteps; paths=zeros(n,3,total);
            velocities=repmat(v,n,1); positions=zeros(n,3); endpoints=zeros(n,3);
            feasible=true(n,1);
            for k=1:total
                desired=commands;
                if k>steps, desired(:)=0; end
                % Initial command and the stop tail each include latency.
                elapsed=k*dt;
                if k>steps, elapsed=(k-steps)*dt; end
                activeDt=max(0,min(dt,elapsed-o.GovernorLatencyS));
                change=desired-velocities;
                duration=min(activeDt,vecnorm(change,2,2)/o.BrakingDecelerationMps2);
                increment=change.*min(1,o.BrakingDecelerationMps2*activeDt./max(vecnorm(change,2,2),1e-12));
                % Integrate coasting, constant acceleration, then target hold.
                positions=positions+velocities*dt+increment.*(activeDt-duration/2);
                velocities=velocities+increment; paths(:,:,k)=positions;
                % Remain within the observed spherical horizon and mission domain.
                feasible=feasible & vecnorm(positions,2,2)<=7-.4-radius & ...
                    all(p+positions>=o.WorldBoundsNedM(:,1)' & p+positions<=o.WorldBoundsNedM(:,2)',2);
                if k==steps, endpoints=positions; end
            end
            queries=reshape(permute(paths,[1 3 2]),[],3);
            nearest=Inf(size(queries,1),1);
            % Bound the distance-matrix allocation for dense reconstructed scans.
            for first=1:512:size(points,1)
                cloud=points(first:min(first+511,end),:);
                d2=max(sum(queries.^2,2)+sum(cloud.^2,2)'-2*queries*cloud',0);
                nearest=min(nearest,min(d2,[],2));
            end
            clearance=sqrt(min(reshape(nearest,n,total),[],2));
            feasible=feasible & clearance>=radius;
        end
    end
end
