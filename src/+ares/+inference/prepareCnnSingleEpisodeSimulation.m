function prepareCnnSingleEpisodeSimulation(projectRoot,episodeId,intermittent)
%PREPARECNNSINGLEEPISODESIMULATION Keep selected start on capture and toolbar Run.
arguments
    projectRoot (1,1) string
    episodeId (1,1) string
    intermittent (1,1) logical
end
initializer=fullfile(projectRoot,"scripts","initializeSingleUav.m");
if intermittent
    cfg=ares.inference.cnnSingleEpisodeConfig(projectRoot,episodeId);
    schedule=ares.inference.createIntermittentSchedule(double(cfg.simulation.stopTime));
    assignin("base","SingleUavConfigOverride",cfg);
    assignin("base","SingleUavLidarScheduleOverride",schedule);
else
    evalin("base","clear SingleUavConfigOverride SingleUavLidarScheduleOverride;");
end
evalin("base","run('"+replace(initializer,"'","''")+"');");
end
