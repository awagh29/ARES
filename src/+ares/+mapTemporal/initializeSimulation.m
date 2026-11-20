function [map,cfg,schedule,intervals] = initializeSimulation(projectRoot,planFile,options)
%INITIALIZESIMULATION Recreate the scene and requested state on every Run.
if nargin<3, saved=load(planFile,'options'); options=saved.options; end
o=ares.mapTemporal.validateOptions(options);
[schedule,intervals]=ares.navigation.outageSchedule(ares.mapTemporal.navigationOptions(o));
cfg=ares.config.defaultSingleUav();
cfg.mission.initialPositionNed=o.StartState(1:3);
cfg.mission.initialEulerZYXRad=o.StartEulerZYXRad;
cfg.mission.waypointsNed=[o.StartState(1:3);o.GoalState(1:3)];
% Give the original follower a finite segment direction for a hover mission.
% The actual initial state and terminal goal remain exactly as requested.
if norm(o.StartState(1:3)-o.GoalState(1:3))<1e-6
    inward=mean(o.WorldBoundsNedM,2)'-o.GoalState(1:3);
    if norm(inward)<1e-6, inward=[1 0 0]; end
    cfg.mission.waypointsNed(1,:)=o.GoalState(1:3)+.01*inward/norm(inward);
end
cfg.simulation.stopTime=o.StopTimeS;
cfg.sensors.lidar.enabled=logical(schedule.Data(1));
cfg.sensors.lidar.mode="map-temporal-"+o.LidarMode;
cfg.vehicle.radiusM=o.VehicleRadiusM;
cfg.safety.minimumSurfaceClearanceM=o.MinimumSurfaceClearanceM;
cfg.acceptance.finalWaypointToleranceM=o.GoalPositionToleranceM;
cfg.acceptance.minimumAltitudeM=-o.WorldBoundsNedM(3,2);
cfg.acceptance.maximumAltitudeM=-o.WorldBoundsNedM(3,1);
assignin('base','SingleUavConfigOverride',cfg);
assignin('base','SingleUavLidarScheduleOverride',schedule);
initializer=fullfile(projectRoot,'scripts','initializeSingleUav.m');
evalin('base',"run('"+replace(string(initializer),"'","''")+"');");
assignin('base','MapTemporalInitialVelocity',o.StartState(4:6)');
assignin('base','MapTemporalInitialEulerZYX',o.StartEulerZYXRad');
assignin('base','MapTemporalInitialBodyRates',o.StartBodyRatesRadps');
map=ares.navigation.sceneMap(evalin('base','ObstaclePositions'), ...
    double(evalin('base','ObstacleWidth')),double(evalin('base','ObstacleHeight')));
if nargin<3
    saved=load(planFile,'map');
    assert(isequal(map.BoxMin,saved.map.BoxMin) && isequal(map.BoxMax,saved.map.BoxMax), ...
        'ARES:TemporalSceneChanged','The obstacle scene changed. Prepare the simulation again.');
end
end
