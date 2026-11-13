function [cfg,source] = denseSaitsSingleEpisodeConfig(projectRoot,episodeId)
%DENSESAITSSINGLEEPISODECONFIG Select the source episode initial position and first waypoint.
arguments
    projectRoot (1,1) string
    episodeId (1,1) string
end
assert(~isempty(regexp(episodeId,'^[A-Za-z0-9-]+$','once')), ...
    "ARES:InvalidEpisodeId","Invalid episode ID.");
path=fullfile(projectRoot,"data","raw",episodeId,"manifest.json");
assert(isfile(path),"ARES:MissingEpisode","Missing episode manifest: %s",path);
source=jsondecode(fileread(path));
assert(string(source.runId)==episodeId,"ARES:EpisodeIdMismatch","Manifest ID differs from selected folder.");
cfg=ares.config.defaultSingleUav();
start=reshape(double(source.configuration.mission.initialPositionNed),1,3);
assert(all(isfinite(start)),"ARES:InvalidStart","Invalid recorded start position.");
goal=double(source.configuration.mission.waypointsNed(end,:));
assert(isequal(goal,cfg.mission.waypointsNed(end,:)),"ARES:DifferentEpisodeGoal", ...
    "This integration varies only the start point; the recorded goal must match the baseline.");
cfg.mission.initialPositionNed=start;
cfg.mission.waypointsNed(1,:)=start;
cfg.sensors.lidar.enabled=true;
cfg.sensors.lidar.mode="intermittent-dense-saits";
end
