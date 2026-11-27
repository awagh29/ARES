function [map,cfg,schedule,intervals] = initializeSimulation(projectRoot,planFile,options)
%INITIALIZESIMULATION Fresh scene/initial state/schedule on every toolbar Run.
if nargin<3
    data=load(planFile,'options'); options=data.options;
end
o=ares.navigation.validateOptions(options);
[schedule,intervals]=ares.navigation.outageSchedule(o);
cfg=ares.config.defaultSingleUav();
cfg.mission.initialPositionNed=o.StartState(1:3);
cfg.mission.initialEulerZYXRad=o.StartEulerZYXRad;
cfg.mission.waypointsNed=[o.StartState(1:3);o.GoalState(1:3)];
cfg.simulation.stopTime=o.StopTimeS;
cfg.sensors.lidar.enabled=logical(schedule.Data(1));
cfg.sensors.lidar.mode="map-correction-"+o.LidarMode;
cfg.vehicle.radiusM=o.VehicleRadiusM;
cfg.safety.minimumSurfaceClearanceM=o.MinimumSurfaceClearanceM;
cfg.acceptance.finalWaypointToleranceM=o.GoalPositionToleranceM;
cfg.acceptance.minimumAltitudeM=-o.WorldBoundsNedM(3,2);
cfg.acceptance.maximumAltitudeM=-o.WorldBoundsNedM(3,1);
assignin('base','SingleUavConfigOverride',cfg);
assignin('base','SingleUavLidarScheduleOverride',schedule);
initializer=fullfile(projectRoot,'scripts','initializeSingleUav.m');
evalin('base',"run('"+replace(string(initializer),"'","''")+"');");
assignin('base','MapCorrectionInitialVelocity',o.StartState(4:6)');
assignin('base','MapCorrectionInitialEulerZYX',o.StartEulerZYXRad');
assignin('base','MapCorrectionInitialBodyRates',o.StartBodyRatesRadps');
map=ares.navigation.sceneMap(evalin('base','ObstaclePositions'), ...
    double(evalin('base','ObstacleWidth')),double(evalin('base','ObstacleHeight')));
if nargin<3
    data=load(planFile,'map');
    assert(isequal(map.BoxMin,data.map.BoxMin) && isequal(map.BoxMax,data.map.BoxMax), ...
        'ARES:MapSceneChanged','The obstacle scene changed. Prepare a new route before running.');
end
end
