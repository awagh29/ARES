function [artifactPath,deployment] = exportCnnSingleEpisodeModel(projectRoot,episodeId,pythonExecutable,trainingRunDirectory)
%EXPORTCNNSINGLEEPISODEMODEL Resolve the matching completed results run by ID.
arguments
    projectRoot (1,1) string
    episodeId (1,1) string
    pythonExecutable (1,1) string
    trainingRunDirectory (1,1) string = ""
end
assert(isfile(pythonExecutable),"ARES:MissingCnnPython","Missing Python executable: %s",pythonExecutable);
ares.inference.cnnSingleEpisodeConfig(projectRoot,episodeId); % Validate ID/start first.
script=fullfile(projectRoot,"scripts","export_cnn_mlp_matlab.py");
command='"'+pythonExecutable+'" "'+script+'" --project-root "'+projectRoot+ ...
    '" --episode-id "'+episodeId+'"';
if strlength(trainingRunDirectory)>0
    command=command+' --training-run "'+trainingRunDirectory+'"';
end
[status,text]=system(command);
assert(status==0,"ARES:CnnExportFailed","CNN export failed: %s",string(text));
pointer=fullfile(projectRoot,"data","models","cnn_mlp_single_episode",episodeId,"deployment.json");
deployment=jsondecode(fileread(pointer));
assert(string(deployment.episode_id)==episodeId,"ARES:WrongCnnEpisode","Deployment belongs to another episode.");
artifactPath=string(deployment.model_path);
assert(isfile(artifactPath) && ares.inference.fileSha256(artifactPath)==string(deployment.model_mat_sha256), ...
    "ARES:CnnArtifactChanged","Exported model hash differs from its manifest.");
fprintf("CNN source checkpoint: %s\nMATLAB model: %s\n",string(deployment.checkpoint_path),artifactPath);
end
