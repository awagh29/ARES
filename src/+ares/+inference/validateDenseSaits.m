function report = validateDenseSaits(artifactPath)
%VALIDATEDENSESAITS Verify exported dense SAITS FP32 fixtures before controller wiring.
arguments
    artifactPath (1,1) string
end
artifact=load(artifactPath,"model","fixtures");
f=artifact.fixtures; errors=zeros(size(f.rawHistory,3),1); elapsed=errors;
for k=1:numel(errors)
    timer=tic;
    ranges=ares.inference.predictDenseSaits(artifact.model,f.rawHistory(:,:,k),logical(f.observedMask(:,:,k)));
    elapsed(k)=toc(timer);
    errors(k)=max(abs(double(ranges)-double(f.predictedRanges(:,:,k))),[],"all");
end
report=struct("schemaVersion","ares-dense-saits-matlab-validation-v1", ...
    "episodeId",string(artifact.model.episodeId),"modelSha256",ares.inference.fileSha256(artifactPath), ...
    "maximumPredictionError",max(errors),"absoluteTolerance",double(f.absoluteTolerance), ...
    "fixtureCount",numel(errors),"medianInferenceMilliseconds",1000*median(elapsed), ...
    "matlabRelease",string(version("-release")),"plantSimulationRun",false);
report.allParityChecksPassed=all(errors<=double(f.absoluteTolerance));
assert(report.allParityChecksPassed,"ARES:DenseSaitsMatlabParityFailed", ...
    "Dense SAITS MATLAB/PyTorch parity failed: %.8g (limit %.8g)",max(errors),double(f.absoluteTolerance));
path=fullfile(fileparts(artifactPath),"matlab_validation.json");
fid=fopen(path,"w","n","UTF-8"); assert(fid~=-1,"ARES:DenseSaitsValidationWriteFailed","Cannot write %s",path);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
clear cleanup; % Publish verification status only after the report is closed.
exportPath=fullfile(fileparts(artifactPath),"export_manifest.json");
if isfile(exportPath)
    exported=jsondecode(fileread(exportPath));
    assert(string(exported.model_mat_sha256)==report.modelSha256, ...
        "ARES:DenseSaitsExportManifestMismatch","Export manifest does not match validated weights.");
    exported.matlab_parity_verified=true;
    exported.matlab_validation_report=path;
    exported.matlab_maximum_prediction_error=report.maximumPredictionError;
    writeStatus(exportPath,exported);
    pointerPath=fullfile(fileparts(fileparts(artifactPath)),"deployment.json");
    if isfile(pointerPath)
        pointer=jsondecode(fileread(pointerPath));
        if string(pointer.model_path)==artifactPath && string(pointer.model_mat_sha256)==report.modelSha256
            pointer.matlab_parity_verified=true;
            pointer.matlab_validation_report=path;
            pointer.matlab_maximum_prediction_error=report.maximumPredictionError;
            writeStatus(pointerPath,pointer);
        end
    end
end
end

function writeStatus(path,value)
temporary=path+".tmp";
fid=fopen(temporary,"w","n","UTF-8"); assert(fid~=-1,"ARES:WriteFailed","Cannot write %s",temporary);
cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
clear cleanup;
movefile(temporary,path,"f");
end
