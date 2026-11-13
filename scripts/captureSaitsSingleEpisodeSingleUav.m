%CAPTURESAITSSINGLEEPISODESINGLEUAV Run-ID selects start and trained dense SAITS only.
% Existing Simulink plant, scene, VFH/PID and seed-513 outages are preserved.
% Set SaitsSingleEpisodeIdOverride for future episodes, or edit the ID below.
project=currentProject;
saitsSingleEpisodeProjectRoot=string(project.RootFolder);
saitsSingleEpisodeEpisodeId="f9ab72ef-e1d7-4479-8faa-03bb5b36776c";
if exist("SaitsSingleEpisodeIdOverride","var"), saitsSingleEpisodeEpisodeId=string(SaitsSingleEpisodeIdOverride); end
saitsSingleEpisodePython=fullfile(saitsSingleEpisodeProjectRoot,".venv","Scripts","python.exe");
if exist("SaitsSingleEpisodePythonExecutableOverride","var"), saitsSingleEpisodePython=string(SaitsSingleEpisodePythonExecutableOverride); end
saitsSingleEpisodeTrainingRun=""; % Empty explicitly selects latest finished results run for this episode.
if exist("SaitsSingleEpisodeTrainingRunDirectoryOverride","var"), saitsSingleEpisodeTrainingRun=string(SaitsSingleEpisodeTrainingRunDirectoryOverride); end
saitsSingleEpisodeShowSimulation=usejava("desktop");
if exist("SaitsSingleEpisodeShowSimulationOverride","var"), saitsSingleEpisodeShowSimulation=logical(SaitsSingleEpisodeShowSimulationOverride); end
[saitsSingleEpisodeArtifact,saitsSingleEpisodeDeployment]=ares.inference.exportDenseSaitsSingleEpisodeModel( ...
    saitsSingleEpisodeProjectRoot,saitsSingleEpisodeEpisodeId,saitsSingleEpisodePython,saitsSingleEpisodeTrainingRun);
saitsSingleEpisodeValidation=ares.inference.validateDenseSaits(saitsSingleEpisodeArtifact);
[saitsSingleEpisodeManifest,saitsSingleEpisodeTelemetry,saitsSingleEpisodeManifestPath,saitsSingleEpisodeTelemetryPath,saitsSingleEpisodeSummary]= ...
    ares.inference.captureDenseSaitsSingleEpisode(saitsSingleEpisodeProjectRoot,saitsSingleEpisodeArtifact,saitsSingleEpisodeShowSimulation,saitsSingleEpisodeEpisodeId);
