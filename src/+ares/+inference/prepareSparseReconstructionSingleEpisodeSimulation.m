function prepareSparseReconstructionSingleEpisodeSimulation(projectRoot,episodeId,intermittent,lidarMode)
%PREPARESPARSERECONSTRUCTIONSINGLEEPISODESIMULATION Restore the selected source start and outage schedule on toolbar Run.
arguments
    projectRoot (1,1) string
    episodeId (1,1) string
    intermittent (1,1) logical
    lidarMode (1,1) string {mustBeMember(lidarMode,["intermittent","disabled"])} = "intermittent"
end
initializer=fullfile(projectRoot,"scripts","initializeSingleUav.m");
if intermittent
    cfg=ares.inference.sparseReconstructionSingleEpisodeConfig(projectRoot,episodeId);
    cfg.sensors.lidar.enabled=lidarMode~="disabled";
    cfg.sensors.lidar.mode=lidarMode+"-sparse-reconstruction";
    schedule=ares.inference.sparseReconstructionLidarScenario(double(cfg.simulation.stopTime),lidarMode);
    assignin("base","SingleUavConfigOverride",cfg);
    assignin("base","SingleUavLidarScheduleOverride",schedule);
else
    evalin("base","clear SingleUavConfigOverride SingleUavLidarScheduleOverride;");
end
evalin("base","run('"+replace(initializer,"'","''")+"');");
end
