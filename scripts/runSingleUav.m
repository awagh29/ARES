proj = currentProject;
projectRoot = proj.RootFolder;

run(fullfile( ...
    projectRoot, ...
    "scripts", ...
    "initializeSingleUav.m"));

modelName = "ares_single_uav_top";
modelPath = fullfile(projectRoot, "models", modelName + ".slx");

load_system(modelPath);

set_param( ...
    modelName, ...
    "StopTime", ...
    num2str(cfg.simulation.stopTime));

disp("Starting ARES single-UAV simulation...");

simOut = sim(modelName);

points = squeeze(simOut.trajectoryPoints(1,:,:))';

finalPosition = points(end,:);
finalWaypoint = Waypoints(end,:);

finalWaypointErrorM = norm(finalPosition - finalWaypoint);
maximumAltitudeM = max(-points(:,3));

requiredObstacleClearanceM = ...
    cfg.vehicle.radiusM ...
    + cfg.safety.minimumObstacleDistanceM;

fprintf("Simulation completed.\n");
fprintf("Final position: [%.2f, %.2f, %.2f] m\n", finalPosition);
fprintf("Final waypoint error: %.2f m\n", finalWaypointErrorM);
fprintf("Maximum altitude: %.2f m\n", maximumAltitudeM);
fprintf("Required obstacle clearance: %.2f m\n", ...
    requiredObstacleClearanceM);


minimumAltitudeM = min(-points(:,3));
%% Calculate minimum horizontal obstacle clearance

horizontalTrajectory = points(:,1:2);
minimumObstacleClearanceM = inf;

for obstacleIndex = 1:size(ObstaclePositions, 1)

    obstacleCenterEastNorth = ObstaclePositions(obstacleIndex,:);
    obstacleCenter = fliplr(obstacleCenterEastNorth);
    obstacleHalfWidth = ObstacleWidth / 2;

    % Distance from each drone position to the obstacle boundary
    delta = abs(horizontalTrajectory - obstacleCenter) ...
        - obstacleHalfWidth;

    outsideDelta = max(delta, 0);

    distanceToObstacle = hypot( ...
        outsideDelta(:,1), ...
        outsideDelta(:,2));

    minimumObstacleClearanceM = min( ...
        minimumObstacleClearanceM, ...
        min(distanceToObstacle));
end

assert( ...
    all(isfinite(points), "all"), ...
    "ARES failure: trajectory contains NaN or Inf values.");

assert( ...
    finalWaypointErrorM <= cfg.acceptance.finalWaypointToleranceM, ...
    "ARES failure: final waypoint error exceeded its limit.");

assert( ...
    minimumAltitudeM >= cfg.acceptance.minimumAltitudeM, ...
    "ARES failure: drone flew below the safe altitude.");

assert( ...
    maximumAltitudeM <= cfg.acceptance.maximumAltitudeM, ...
    "ARES failure: drone exceeded the safe altitude.");

assert(minimumObstacleClearanceM >= requiredObstacleClearanceM, ...
    "ARES collision-safety failure: actual clearance %.2f m is below required clearance %.2f m.", ...
    minimumObstacleClearanceM, ...
    requiredObstacleClearanceM);

fprintf("Minimum obstacle clearance: %.2f m\n", minimumObstacleClearanceM);


disp("PASS: All nominal mission safety checks passed.");