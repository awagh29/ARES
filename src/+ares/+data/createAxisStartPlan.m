function plan = createAxisStartPlan(seed, cfg, obstacleCentersNed_m, obstacleWidthM, obstacleHeightM)
%CREATEAXISSTARTPLAN Reproducible 40 North + 40 East LiDAR-on starts.
% Geometry defaults snapshot scripts/initializeSingleUav.m: the addMesh
% inputs [10 0;15 15;10 20;5 15;22 18;15 25;10 20] are ENU, so
% their horizontal coordinates are swapped here to match NED positions.
% Sampling is stratified over the cumulative length of safe axis intervals.
% This private random stream does not modify the caller's RNG state.
arguments
    seed (1,1) double {mustBeInteger, mustBeNonnegative} = 20261102
    cfg (1,1) struct = ares.config.defaultSingleUav()
    obstacleCentersNed_m (:,:) double {mustBeFinite} = ...
        [0 10; 15 15; 20 10; 15 5; 18 22; 25 15; 20 10]
    obstacleWidthM (1,1) double {mustBePositive, mustBeFinite} = 3
    obstacleHeightM (1,1) double {mustBePositive, mustBeFinite} = 15
end
assert(seed <= 2^32-1, "ARES:InvalidSeed", "Seed must be at most 2^32-1.");
assert(ismember(size(obstacleCentersNed_m,2),[2 3]), ...
    "ARES:InvalidObstacleCenters", "Obstacle centers must have 2 or 3 columns.");
goal = double(cfg.mission.waypointsNed(end,:));
down = double(cfg.mission.initialPositionNed(3));
assert(all(isfinite(goal)) && numel(goal)==3 && ...
    all(goal(1:2)>0) && isfinite(down), ...
    "ARES:InvalidAxisBounds", "The current scenario requires positive N/E goal spans.");
requiredCenterClearance = double(cfg.vehicle.radiusM) + ...
    double(cfg.safety.minimumSurfaceClearanceM);
assert(isfinite(requiredCenterClearance) && requiredCenterClearance>0, ...
    "ARES:InvalidClearance", "Required center clearance must be positive.");
if size(obstacleCentersNed_m,2)==2
    centers = [obstacleCentersNed_m, ...
        repmat(-obstacleHeightM/2,size(obstacleCentersNed_m,1),1)];
else
    centers = obstacleCentersNed_m;
end
stream = RandStream("mt19937ar", "Seed", seed);
coordinates = zeros(40,2);
for axisIndex = 1:2
    intervals = [0 goal(axisIndex)];
    otherAxis = 3-axisIndex;
    for obstacleIndex = 1:size(centers,1)
        transverseDistance = max(abs(centers(obstacleIndex,otherAxis)) ...
            - obstacleWidthM/2,0);
        verticalDistance = max(abs(down-centers(obstacleIndex,3)) ...
            - obstacleHeightM/2,0);
        remainingSquared = requiredCenterClearance^2 ...
            - transverseDistance^2 - verticalDistance^2;
        if remainingSquared<=0
            continue;
        end
        extension = sqrt(remainingSquared);
        exclusion = centers(obstacleIndex,axisIndex) + ...
            [-obstacleWidthM/2-extension, obstacleWidthM/2+extension];
        intervals = subtractInterval(intervals,exclusion);
    end
    lengths = intervals(:,2)-intervals(:,1);
    totalLength = sum(lengths);
    assert(totalLength>0, "ARES:NoSafeAxisStarts", ...
        "The requested axis contains no safe starting interval.");
    jitter = max(eps,rand(stream,40,1));
    cumulativePosition = ((0:39)' + jitter)/40 * totalLength;
    cumulativeEnd = cumsum(lengths);
    for startIndex = 1:40
        intervalIndex = find(cumulativePosition(startIndex)<cumulativeEnd,1);
        priorLength = cumulativeEnd(intervalIndex)-lengths(intervalIndex);
        coordinates(startIndex,axisIndex) = intervals(intervalIndex,1) ...
            + cumulativePosition(startIndex)-priorLength;
    end
end
RunIndex = (1:80)';
Axis = [repmat("North",40,1); repmat("East",40,1)];
AxisIndex = repmat((1:40)',2,1);
PositionNorthM = [coordinates(:,1); zeros(40,1)];
PositionEastM = [zeros(40,1); coordinates(:,2)];
PositionDownM = repmat(down,80,1);
SimulationSeed = mod(seed+RunIndex,2^32);
plan = table(RunIndex,Axis,AxisIndex,PositionNorthM,PositionEastM, ...
    PositionDownM,SimulationSeed);
plan.Properties.Description = ...
    "80 LiDAR-on starts, stratified randomly on positive NED axes; mesh geometry preserved.";
end

function remaining = subtractInterval(intervals, exclusion)
remaining = zeros(0,2);
for index = 1:size(intervals,1)
    current = intervals(index,:);
    if exclusion(2)<=current(1) || exclusion(1)>=current(2)
        remaining(end+1,:) = current; %#ok<AGROW>
        continue;
    end
    if exclusion(1)>current(1)
        remaining(end+1,:) = [current(1),min(exclusion(1),current(2))]; %#ok<AGROW>
    end
    if exclusion(2)<current(2)
        remaining(end+1,:) = [max(exclusion(2),current(1)),current(2)]; %#ok<AGROW>
    end
end
end
