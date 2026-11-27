classdef ObservedRouteGuide < handle
    % A 2-D route-cost guide from selected scans, not the supplied truth map.
    % Unknown cells are expensive guidance hypotheses; only the rollout
    % checker can admit an executed trajectory through currently observed space.
    properties (Access=private)
        Options
        North
        East
        Positions
        Occupied
        Seen
        Costs
        Edges
        Tick = 0
        Resolution = .5
    end
    methods
        function obj=ObservedRouteGuide(options)
            obj.Options=options;
            obj.North=options.WorldBoundsNedM(1,1):obj.Resolution:options.WorldBoundsNedM(1,2);
            obj.East=options.WorldBoundsNedM(2,1):obj.Resolution:options.WorldBoundsNedM(2,2);
            [n,e]=ndgrid(obj.North,obj.East); obj.Positions=[n(:),e(:)];
            grid=reshape(1:numel(n),size(n)); edges=zeros(0,4);
            for d=[1 0;0 1;1 1;1 -1]'
                ni=1:size(n,1)-d(1); ei=max(1,1-d(2)):min(size(n,2),size(n,2)-d(2));
                a=grid(ni,ei); b=grid(ni+d(1),ei+d(2));
                c=grid(ni+d(1),ei); f=grid(ni,ei+d(2));
                edges=[edges;[a(:),b(:),c(:),f(:)]]; %#ok<AGROW>
            end
            obj.Edges=edges; obj.reset();
        end
        function reset(obj)
            obj.Occupied=false(numel(obj.North),numel(obj.East)); obj.Seen=obj.Occupied;
            obj.Costs=Inf(numel(obj.Occupied),1); obj.Tick=0;
        end
        function costs=step(obj,p,offsets,queries)
            o=obj.Options; obj.Tick=obj.Tick+1;
            % Route observations describe the requested altitude cross-section.
            points=offsets(abs(p(3)+offsets(:,3)-o.GoalState(3))<.65,1:2);
            radius=vecnorm(points,2,2);
            bins=1+mod(floor(atan2(points(:,2),points(:,1))*72/(2*pi)),72);
            limits=accumarray(bins,radius,[72 1],@min,7);
            if abs(p(3)-o.GoalState(3))<.65
                delta=obj.Positions-p(1:2); range=vecnorm(delta,2,2);
                cellBins=1+mod(floor(atan2(delta(:,2),delta(:,1))*72/(2*pi)),72);
                free=range<min(6.4,limits(cellBins)-.4);
                obj.Seen(free)=true; obj.Occupied(free)=false;
            end
            % Do not retain near-max-range reconstruction rings as obstacles.
            points=points(radius<5.5,:)+p(1:2);
            index=indices(obj,points); index=index(index>0);
            obj.Occupied(index)=true; obj.Seen(index)=true;
            if mod(obj.Tick-1,10)==0
                r=o.VehicleRadiusM+o.MinimumSurfaceClearanceM+o.GovernorMarginM+sqrt(2)*obj.Resolution/2;
                extent=ceil(r/obj.Resolution); [x,y]=ndgrid(-extent:extent);
                kernel=(x.^2+y.^2)*obj.Resolution^2<=r^2;
                blocked=conv2(double(obj.Occupied),double(kernel),'same')>0;
                goal=indices(obj,o.GoalState(1:2)); blocked(goal)=false;
                edge=obj.Edges;
                keep=~blocked(edge(:,1)) & ~blocked(edge(:,2)) & ~blocked(edge(:,3)) & ~blocked(edge(:,4));
                edge=edge(keep,:);
                cost=ones(numel(blocked),1); cost(~obj.Seen)=3;
                length=vecnorm(obj.Positions(edge(:,1),:)-obj.Positions(edge(:,2),:),2,2);
                weight=length.*(cost(edge(:,1))+cost(edge(:,2)))/2;
                network=graph(edge(:,1),edge(:,2),weight,numel(blocked));
                obj.Costs=distances(network,goal)';
            end
            query=indices(obj,queries(:,1:2)); costs=Inf(size(query));
            keep=query>0; costs(keep)=obj.Costs(query(keep));
        end
    end
    methods (Access=private)
        function index=indices(obj,points)
            n=round((points(:,1)-obj.North(1))/obj.Resolution)+1;
            e=round((points(:,2)-obj.East(1))/obj.Resolution)+1;
            valid=n>=1 & n<=numel(obj.North) & e>=1 & e<=numel(obj.East);
            index=zeros(size(n)); index(valid)=n(valid)+(e(valid)-1)*numel(obj.North);
        end
    end
end
