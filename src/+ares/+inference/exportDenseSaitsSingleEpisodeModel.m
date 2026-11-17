function [artifactPath,deployment] = exportDenseSaitsSingleEpisodeModel(projectRoot,episodeId,pythonExecutable,trainingRunDirectory)
%EXPORTDENSESAITSSINGLEEPISODEMODEL Resolve and export the selected completed dense SAITS stage-one run.
arguments
    projectRoot (1,1) string
    episodeId (1,1) string
    pythonExecutable (1,1) string
    trainingRunDirectory (1,1) string = ""
end
assert(isfile(pythonExecutable),"ARES:MissingDenseSaitsPython","Missing Python executable: %s",pythonExecutable);
ares.inference.denseSaitsSingleEpisodeConfig(projectRoot,episodeId); % Validate ID/start first.
script=fullfile(projectRoot,"scripts","export_dense_saits_matlab.py");
command='"'+pythonExecutable+'" "'+script+'" --project-root "'+projectRoot+ ...
    '" --episode-id "'+episodeId+'"';
if strlength(trainingRunDirectory)>0
    command=command+' --training-run "'+trainingRunDirectory+'"';
end
[status,text]=system(command);
assert(status==0,"ARES:DenseSaitsExportFailed","DENSE SAITS export failed: %s",string(text));
pointer=fullfile(projectRoot,"data","models","saits_dense_single_episode",episodeId,"deployment.json");
deployment=jsondecode(fileread(pointer));
assert(string(deployment.episode_id)==episodeId,"ARES:WrongDenseSaitsEpisode","Deployment belongs to another episode.");
artifactPath=string(deployment.model_path);
assert(isfile(artifactPath) && ares.inference.fileSha256(artifactPath)==string(deployment.model_mat_sha256), ...
    "ARES:DenseSaitsArtifactChanged","Exported model hash differs from its manifest.");
fprintf("DENSE SAITS source checkpoint: %s\nMATLAB model: %s\n",string(deployment.checkpoint_path),artifactPath);
end
