function report = verifyCnnMlpSingleEpisode(projectRoot,episodeId)
%VERIFYCNNMLPSINGLEEPISODE Read-only newest/deployed/flown model verification.
arguments
    projectRoot (1,1) string
    episodeId (1,1) string
end
python=fullfile(projectRoot,".venv","Scripts","python.exe");
script=fullfile(projectRoot,"scripts","export_cnn_mlp_matlab.py");
command='"'+python+'" "'+script+'" --project-root "'+projectRoot+ ...
    '" --episode-id "'+episodeId+'" --verify-integration';
[status,text]=system(command);
assert(status==0,"ARES:CnnVerificationFailed","Verification failed: %s",string(text));
report=jsondecode(text);
fprintf("Latest completed training run:\n%s\n",string(report.latest_training_run));
fprintf("Selected checkpoint: best.pt, epoch %d\nSHA-256: %s\n",report.best_epoch,string(report.checkpoint_sha256));
fprintf("Deployment matches latest: %d\nMATLAB numerical parity verified: %d\n", ...
    report.deployment_matches_latest,report.matlab_parity_verified);
fprintf("Latest completed capture used latest model: %d\nScan CSV agrees with manifest: %d\n", ...
    report.latest_capture_used_latest,report.capture_logging_verified);
if isfield(report,"reconstruction_used_scans")
    fprintf("Reconstruction used: %d/%d scans; controller influenced: %d scans\n", ...
        report.reconstruction_used_scans,report.total_scans,report.controller_influenced_scans);
end
model="ares_single_uav_cnn_"+erase(episodeId,"-");
report.live_model_loaded=bdIsLoaded(model);
if report.live_model_loaded
    block=model+"/ARES CNN Reconstruction";
    if getSimulinkBlockHandle(block)~=-1 && isfield(report,"model_path")
        report.live_block_artifact=string(get_param(block,"ArtifactPath"));
        report.live_block_matches_deployment=strcmpi(report.live_block_artifact,string(report.model_path));
        fprintf("Open diagram's artifact matches deployment: %d\n",report.live_block_matches_deployment);
    end
end
end
