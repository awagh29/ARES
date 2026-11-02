function metrics = evaluateAxisRun(telemetry, cfg, obstacleCentersNed_m, obstacleWidthM, obstacleHeightM)
%EVALUATEAXISRUN Evaluate every native position sample without discarding failures.
% Nx2 obstacles have horizontal NED centers and Down bounds [-height,0].
% Nx3 obstacles specify explicit NED box centers. Distances are signed:
% center-to-mesh distance minus the configured spherical vehicle radius.
% Collision flags describe sampled positions, not a continuous collision proof.
arguments
    telemetry (1,1) struct
    cfg (1,1) struct
    obstacleCentersNed_m (:,:) double {mustBeFinite}
    obstacleWidthM (1,1) double {mustBePositive, mustBeFinite} = 3
    obstacleHeightM (1,1) double {mustBePositive, mustBeFinite} = 15
end
assert(ismember(size(obstacleCentersNed_m,2),[2 3]), ...
    "ARES:InvalidObstacleCenters", "Obstacle centers must have 2 or 3 columns.");
signal = telemetry.signals.position_ned_m;
time = double(signal.time_s(:));
position = double(signal.values);
metrics.positionSampleCount = numel(time);
metrics.telemetryValid = size(position,1)==numel(time) && ...
    size(position,2)==3 && numel(time)>=2 && ...
    all(isfinite(time)) && all(isfinite(position),"all") && ...
    all(diff(time)>=0);
metrics.plannedInitialPositionNedM = double(cfg.mission.initialPositionNed);
metrics.goalPositionNedM = double(cfg.mission.waypointsNed(end,:));
metrics.actualInitialPositionNedM = [NaN NaN NaN];
metrics.finalPositionNedM = [NaN NaN NaN];
metrics.initialPositionErrorM = NaN;
metrics.initialPositionMatchesPlan = false;
metrics.finalWaypointErrorM = NaN;
metrics.minimumAltitudeM = NaN;
metrics.maximumAltitudeM = NaN;
metrics.minimumObstacleCenterDistanceM = NaN;
metrics.minimumObstacleClearanceM = NaN;
metrics.requiredObstacleClearanceM = double(cfg.safety.minimumSurfaceClearanceM);
metrics.requiredObstacleCenterDistanceM = metrics.requiredObstacleClearanceM ...
    + double(cfg.vehicle.radiusM);
metrics.collisionOccurred = false;
metrics.nearCollisionOccurred = false;
metrics.altitudeWithinLimits = false;
metrics.goalReachedAtAnyTime = false;
metrics.goalReachedAtTermination = false;
metrics.firstGoalEntryTimeS = NaN;
metrics.durationS = NaN;
metrics.finalTimeS = NaN;
metrics.reachedTimeLimit = false;
metrics.timedOut = false;
metrics.totalPathLengthM = NaN;
metrics.safetyChecksPassed = false;
metrics.missionSuccess = false;
metrics.terminationReason = "invalid-telemetry";
metrics.clearanceEvaluation = "signed-3d-box-distance-at-native-position-samples";
if ~metrics.telemetryValid
    return;
end
metrics.actualInitialPositionNedM = position(1,:);
metrics.finalPositionNedM = position(end,:);
metrics.initialPositionErrorM = norm(position(1,:) ...
    - metrics.plannedInitialPositionNedM);
% The Guidance Model stores its initial state as single precision.
metrics.initialPositionMatchesPlan = metrics.initialPositionErrorM<=1e-4;
distanceToGoal = vecnorm(position-metrics.goalPositionNedM,2,2);
inGoal = distanceToGoal<=double(cfg.acceptance.finalWaypointToleranceM);
metrics.finalWaypointErrorM = distanceToGoal(end);
metrics.goalReachedAtAnyTime = any(inGoal);
metrics.goalReachedAtTermination = inGoal(end);
if any(inGoal)
    metrics.firstGoalEntryTimeS = time(find(inGoal,1));
end
altitude = -position(:,3);
metrics.minimumAltitudeM = min(altitude);
metrics.maximumAltitudeM = max(altitude);
metrics.altitudeWithinLimits = ...
    metrics.minimumAltitudeM>=double(cfg.acceptance.minimumAltitudeM) && ...
    metrics.maximumAltitudeM<=double(cfg.acceptance.maximumAltitudeM);
if size(obstacleCentersNed_m,2)==2
    centers = [obstacleCentersNed_m, ...
        repmat(-obstacleHeightM/2,size(obstacleCentersNed_m,1),1)];
else
    centers = obstacleCentersNed_m;
end
minimumSignedDistance = inf;
halfExtents = [obstacleWidthM/2 obstacleWidthM/2 obstacleHeightM/2];
for obstacleIndex = 1:size(centers,1)
    delta = abs(position-centers(obstacleIndex,:))-halfExtents;
    outsideDistance = vecnorm(max(delta,0),2,2);
    insideDistance = min(max(delta,[],2),0);
    minimumSignedDistance = min(minimumSignedDistance, ...
        min(outsideDistance+insideDistance));
end
metrics.minimumObstacleCenterDistanceM = minimumSignedDistance;
metrics.minimumObstacleClearanceM = minimumSignedDistance-double(cfg.vehicle.radiusM);
metrics.collisionOccurred = metrics.minimumObstacleClearanceM<=0;
metrics.nearCollisionOccurred = ...
    metrics.minimumObstacleClearanceM<metrics.requiredObstacleClearanceM;
metrics.durationS = time(end)-time(1);
metrics.finalTimeS = time(end);
metrics.totalPathLengthM = sum(vecnorm(diff(position,1,1),2,2));
timeTolerance = max(1e-6,double(cfg.simulation.sampleTime));
metrics.reachedTimeLimit = time(end)>=double(cfg.simulation.stopTime)-timeTolerance;
metrics.timedOut = metrics.reachedTimeLimit && ~metrics.goalReachedAtTermination;
metrics.safetyChecksPassed = metrics.altitudeWithinLimits && ...
    ~metrics.nearCollisionOccurred;
metrics.missionSuccess = metrics.goalReachedAtTermination && ...
    metrics.safetyChecksPassed && metrics.initialPositionMatchesPlan;
if metrics.goalReachedAtTermination
    metrics.terminationReason = "goal-reached";
elseif metrics.timedOut
    metrics.terminationReason = "time-limit";
else
    metrics.terminationReason = "early-stop-before-goal";
end
end
