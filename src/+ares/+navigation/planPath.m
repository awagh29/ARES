function plan = planPath(map,options)
%PLANPATH 26-connected 3-D A*, positive clearance cost, checked shortcuts.
o=ares.navigation.validateOptions(options);
start=o.StartState(1:3); goal=o.GoalState(1:3); bounds=o.WorldBoundsNedM;
radius=o.VehicleRadiusM+o.MinimumSurfaceClearanceM+o.GeometryUncertaintyM+o.TrackingMarginM;
padding=radius+o.PlanningExtraMarginM;
for point=[start;goal]'
    assert(ares.navigation.segmentFree(point',point',map,padding,bounds), ...
        'ARES:MapEndpointBlocked','Start/goal is outside bounds or inside a planning-padded obstacle.');
end
if ares.navigation.segmentFree(start,goal,map,padding,bounds)
    route=[start;goal]; expanded=0;
else
    step=o.GridResolutionM;
    axes=cell(1,3);
    for k=1:3, axes{k}=bounds(k,1):step:bounds(k,2); end
    shape=cellfun(@numel,axes); count=prod(shape);
    assert(count<=1e6,'ARES:MapGridTooLarge','Grid exceeds one million cells; increase GridResolutionM or tighten bounds.');
    [north,east,down]=ndgrid(axes{:}); nodes=[north(:),east(:),down(:)];
    occupied=false(count,1);
    for k=1:size(map.BoxMin,1)
        occupied=occupied | all(nodes>=map.BoxMin(k,:)-padding & nodes<=map.BoxMax(k,:)+padding,2);
    end
    source=anchor(start); target=anchor(goal);
    cost=Inf(count,1); parent=zeros(count,1); closed=false(count,1);
    cost(source)=0; expanded=0;
    heapIds=zeros(max(1024,count),1); heapKeys=Inf(size(heapIds)); heapSize=0;
    push(source,norm(nodes(source,:)-nodes(target,:)));
    [a,b,c]=ndgrid(-1:1,-1:1,-1:1); offsets=[a(:),b(:),c(:)]; offsets(all(offsets==0,2),:)=[];
    while heapSize>0
        current=pop();
        if closed(current), continue; end
        closed(current)=true; expanded=expanded+1;
        if current==target, break; end
        [i,j,k]=ind2sub(shape,current); subs=[i j k]+offsets;
        valid=all(subs>=1 & subs<=shape,2); subs=subs(valid,:);
        neighbors=sub2ind(shape,subs(:,1),subs(:,2),subs(:,3));
        neighbors=neighbors(~occupied(neighbors) & ~closed(neighbors));
        for next=neighbors'
            % Checking the full edge prevents diagonal corner cutting.
            if ~ares.navigation.segmentFree(nodes(current,:),nodes(next,:),map,padding,bounds), continue; end
            trial=cost(current)+edgeCost(nodes(current,:),nodes(next,:));
            if trial<cost(next)
                cost(next)=trial; parent(next)=current;
                push(next,trial+norm(nodes(next,:)-nodes(target,:)));
            end
        end
    end
    assert(closed(target),'ARES:MapNoRoute','No route in the configured 3-D grid. Change bounds/resolution or mission.');
    ids=target;
    while ids(1)~=source, ids=[parent(ids(1));ids]; end %#ok<AGROW>
    route=[start;nodes(ids,:);goal];
end
route=route([true;vecnorm(diff(route),2,2)>1e-10],:);
if size(route,1)==1, route=[route;route]; end
rawRoute=route;
% Keep a shortcut only when its sampled weighted cost does not increase.
result=route(1,:); cursor=1;
while cursor<size(route,1)
    selected=cursor+1; oldCost=0;
    for last=cursor+1:size(route,1)
        oldCost=oldCost+edgeCost(route(last-1,:),route(last,:));
        if ares.navigation.segmentFree(route(cursor,:),route(last,:),map,padding,bounds) && ...
                edgeCost(route(cursor,:),route(last,:))<=oldCost+1e-9
            selected=last;
        end
    end
    result=[result;route(selected,:)]; cursor=selected; %#ok<AGROW>
end
plan.WaypointsNedM=result;
plan.RawWaypointsNedM=rawRoute;
plan.LengthM=sum(vecnorm(diff(result),2,2));
plan.ExpandedNodes=expanded;
plan.SafetyRadiusM=radius;
plan.PlanningPaddingM=padding;
plan.Planner="26-connected 3-D A* with clearance-weighted edge costs and collision-checked shortcuts";
plan.Optimality="A* graph optimum before shortcuts; geometric approximation, not globally time-optimal flight";

    function id=anchor(point)
        squared=sum((nodes-point).^2,2); squared(occupied)=Inf;
        [~,order]=sort(squared);
        id=0;
        for candidate=order(1:min(200,numel(order)))'
            if isfinite(squared(candidate)) && ares.navigation.segmentFree(point,nodes(candidate,:),map,padding,bounds)
                id=candidate; break;
            end
        end
        assert(id>0,'ARES:MapGridAnchor','No collision-free grid connection for start/goal. Use a finer grid.');
    end
    function value=edgeCost(first,last)
        lengthM=norm(last-first);
        samples=max(2,ceil(lengthM/(o.GridResolutionM/2))+1);
        values=zeros(samples,1);
        for index=1:samples
            p=first+(last-first)*(index-1)/(samples-1);
            clearance=max(0,min(ares.navigation.mapClearance(p,map))-radius);
            values(index)=1+o.ClearanceWeight/(1+clearance)^2;
        end
        value=lengthM*mean(values);
    end
    function push(id,key)
        heapSize=heapSize+1; slot=heapSize;
        while slot>1
            up=floor(slot/2);
            if heapKeys(up)<=key, break; end
            heapIds(slot)=heapIds(up); heapKeys(slot)=heapKeys(up); slot=up;
        end
        heapIds(slot)=id; heapKeys(slot)=key;
    end
    function id=pop()
        id=heapIds(1); lastId=heapIds(heapSize); key=heapKeys(heapSize); heapSize=heapSize-1;
        slot=1;
        while 2*slot<=heapSize
            child=2*slot;
            if child<heapSize && heapKeys(child+1)<heapKeys(child), child=child+1; end
            if key<=heapKeys(child), break; end
            heapIds(slot)=heapIds(child); heapKeys(slot)=heapKeys(child); slot=child;
        end
        heapIds(slot)=lastId; heapKeys(slot)=key;
    end
end
