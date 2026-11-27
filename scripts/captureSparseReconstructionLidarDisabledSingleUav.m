%CAPTURESPARSERECONSTRUCTIONLIDARDISABLEDSINGLEUAV Single-episode sparse CNN/FiLM; no LiDAR from t=0.
% Same source-episode, Python, training-run and visualization overrides as
% captureSparseSingleEpisodeSingleUav.m. There is NO healthy warm-up or seed scan.
project=currentProject;
sparseSingleEpisodeProjectRoot=string(project.RootFolder);
sparseSingleEpisodeEpisodeId="f9ab72ef-e1d7-4479-8faa-03bb5b36776c";
if exist("SparseSingleEpisodeIdOverride","var"), sparseSingleEpisodeEpisodeId=string(SparseSingleEpisodeIdOverride); end
sparseSingleEpisodePython=fullfile(sparseSingleEpisodeProjectRoot,".venv","Scripts","python.exe");
if exist("SparseSingleEpisodePythonExecutableOverride","var"), sparseSingleEpisodePython=string(SparseSingleEpisodePythonExecutableOverride); end
sparseSingleEpisodeTrainingRun="";
if exist("SparseSingleEpisodeTrainingRunDirectoryOverride","var"), sparseSingleEpisodeTrainingRun=string(SparseSingleEpisodeTrainingRunDirectoryOverride); end
sparseSingleEpisodeShowSimulation=usejava("desktop");
if exist("SparseSingleEpisodeShowSimulationOverride","var"), sparseSingleEpisodeShowSimulation=logical(SparseSingleEpisodeShowSimulationOverride); end
[sparseSingleEpisodeArtifact,sparseSingleEpisodeDeployment]=ares.inference.exportSparseReconstructionSingleEpisodeModel( ...
    sparseSingleEpisodeProjectRoot,sparseSingleEpisodeEpisodeId,sparseSingleEpisodePython,sparseSingleEpisodeTrainingRun);
sparseSingleEpisodeValidation=ares.inference.validateSparseReconstruction(sparseSingleEpisodeArtifact);
[sparseSingleEpisodeManifest,sparseSingleEpisodeTelemetry,sparseSingleEpisodeManifestPath,sparseSingleEpisodeTelemetryPath,sparseSingleEpisodeSummary]= ...
    ares.inference.captureSparseReconstructionSingleEpisode(sparseSingleEpisodeProjectRoot,sparseSingleEpisodeArtifact, ...
        sparseSingleEpisodeShowSimulation,sparseSingleEpisodeEpisodeId,"disabled");
