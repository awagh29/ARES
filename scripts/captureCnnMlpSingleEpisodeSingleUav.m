%CAPTURECNNMLPSINGLEEPISODESINGLEUAV Run-ID selects start and trained CNN only.
% Existing Simulink plant, scene, VFH/PID and seed-513 outages are preserved.
% Set CnnEpisodeIdOverride for future episodes, or edit the ID below.
project=currentProject;
cnnProjectRoot=string(project.RootFolder);
cnnEpisodeId="f9ab72ef-e1d7-4479-8faa-03bb5b36776c";
if exist("CnnEpisodeIdOverride","var"), cnnEpisodeId=string(CnnEpisodeIdOverride); end
cnnPython=fullfile(cnnProjectRoot,".venv","Scripts","python.exe");
if exist("CnnPythonExecutableOverride","var"), cnnPython=string(CnnPythonExecutableOverride); end
cnnTrainingRun=""; % Empty explicitly selects latest finished results run for this episode.
if exist("CnnTrainingRunDirectoryOverride","var"), cnnTrainingRun=string(CnnTrainingRunDirectoryOverride); end
cnnShowSimulation=usejava("desktop");
if exist("CnnShowSimulationOverride","var"), cnnShowSimulation=logical(CnnShowSimulationOverride); end
[cnnArtifact,cnnDeployment]=ares.inference.exportCnnSingleEpisodeModel( ...
    cnnProjectRoot,cnnEpisodeId,cnnPython,cnnTrainingRun);
cnnValidation=ares.inference.validateCnnMlp(cnnArtifact);
[cnnManifest,cnnTelemetry,cnnManifestPath,cnnTelemetryPath,cnnSummary]= ...
    ares.inference.captureCnnSingleEpisode(cnnProjectRoot,cnnArtifact,cnnShowSimulation,cnnEpisodeId);
