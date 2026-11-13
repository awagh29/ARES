function prepareDenseSaitsSingleEpisodeSimulation(projectRoot,episodeId,intermittent,lidarMode)
%PREPAREDENSESAITSSINGLEEPISODESIMULATION Restore the selected source start and outage schedule on toolbar Run.
arguments
    projectRoot (1,1) string
    episodeId (1,1) string
    intermittent (1,1) logical
    lidarMode (1,1) string {mustBeMember(lidarMode,["intermittent","disabled"])} = "intermittent"
end
initializer=fullfile(projectRoot,"scripts","initializeSingleUav.m");
if intermittent
    cfg=ares.inference.denseSaitsSingleEpisodeConfig(projectRoot,episodeId);
    cfg.sensors.lidar.enabled=lidarMode~="disabled";
    cfg.sensors.lidar.mode=lidarMode+"-dense-saits";
    schedule=ares.inference.denseSaitsLidarScenario(double(cfg.simulation.stopTime),lidarMode);
    assignin("base","SingleUavConfigOverride",cfg);
    assignin("base","SingleUavLidarScheduleOverride",schedule);
else
    evalin("base","clear SingleUavConfigOverride SingleUavLidarScheduleOverride;");
end
evalin("base","run('"+replace(initializer,"'","''")+"');");
end
