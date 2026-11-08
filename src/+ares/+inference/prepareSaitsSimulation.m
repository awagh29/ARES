function prepareSaitsSimulation(projectRoot, intermittent)
%PREPARESAITSSIMULATION Initialize fresh scene/state for capture or toolbar Run.
arguments
    projectRoot (1,1) string
    intermittent (1,1) logical = true
end
initializer = fullfile(projectRoot,"scripts","initializeSingleUav.m");
if intermittent
    cfg = ares.config.defaultSingleUav();
    cfg.sensors.lidar.enabled = true;
    cfg.sensors.lidar.mode = "intermittent-saits";
    schedule = ares.inference.createIntermittentSchedule(double(cfg.simulation.stopTime));
    assignin("base","SingleUavConfigOverride",cfg);
    assignin("base","SingleUavLidarScheduleOverride",schedule);
else
    evalin("base","clear SingleUavConfigOverride SingleUavLidarScheduleOverride;");
end
evalin("base","run('"+replace(initializer,"'","''")+"');");
end
