function [artifactPath,deployment] = exportSparseReconstructionSingleEpisodeModel(projectRoot,episodeId,pythonExecutable,trainingRunDirectory)
%EXPORTSPARSERECONSTRUCTIONSINGLEEPISODEMODEL Resolve and export the selected completed sparse CNN/FiLM stage-one run.
arguments
    projectRoot (1,1) string
    episodeId (1,1) string
    pythonExecutable (1,1) string
    trainingRunDirectory (1,1) string = ""
end
assert(isfile(pythonExecutable),"ARES:MissingSparseReconstructionPython","Missing Python executable: %s",pythonExecutable);
ares.inference.sparseReconstructionSingleEpisodeConfig(projectRoot,episodeId); % Validate ID/start first.
script=fullfile(projectRoot,"scripts","export_sparse_reconstruction_matlab.py");
command='"'+pythonExecutable+'" "'+script+'" --project-root "'+projectRoot+ ...
    '" --episode-id "'+episodeId+'"';
if strlength(trainingRunDirectory)>0
    command=command+' --training-run "'+trainingRunDirectory+'"';
end
[status,text]=system(command);
assert(status==0,"ARES:SparseReconstructionExportFailed","Sparse reconstruction export failed: %s",string(text));
pointer=fullfile(projectRoot,"data","models","sparse_reconstruction_single_episode",episodeId,"deployment.json");
deployment=jsondecode(fileread(pointer));
assert(string(deployment.episode_id)==episodeId,"ARES:WrongSparseReconstructionEpisode","Deployment belongs to another episode.");
artifactPath=string(deployment.model_path);
assert(isfile(artifactPath) && ares.inference.fileSha256(artifactPath)==string(deployment.model_mat_sha256), ...
    "ARES:SparseReconstructionArtifactChanged","Exported model hash differs from its manifest.");
fprintf("Sparse reconstruction source checkpoint: %s\nMATLAB model: %s\n",string(deployment.checkpoint_path),artifactPath);
end
