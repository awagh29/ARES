function prepared = prepareSaitsBarrierModel(projectRoot,deploymentPath,pythonExecutable,showSimulation,continueAfterFailure)
%PREPARESAITSBARRIERMODEL Validate, wire and compile a saved experiment copy.
arguments
    projectRoot (1,1) string
    deploymentPath (1,1) string
    pythonExecutable (1,1) string
    showSimulation (1,1) logical = true
    continueAfterFailure (1,1) logical = false
end
assert(isfolder(deploymentPath) && isfile(fullfile(deploymentPath,"deployment.json")), ...
    "ARES:MissingBarrierDeployment","Expected a complete deployment folder: %s",deploymentPath);
deployment=jsondecode(fileread(fullfile(deploymentPath,"deployment.json")));
assert(string(deployment.schema_version)=="ares-augmented-saits-barrier-deployment-v1", ...
    "ARES:WrongBarrierDeployment","Expected the augmented deployment schema.");
cfg=ares.config.defaultSingleUav(); cfg.sensors.lidar.enabled=true; cfg.sensors.lidar.mode="intermittent-saits-barrier";
model="ares_single_uav_saits_barrier";
folder=fullfile(projectRoot,"results","saits_barrier_integration");
if ~isfolder(folder), mkdir(folder); end
modelPath=fullfile(folder,model+".slx"); metadataPath=fullfile(folder,"prepared.json");
if bdIsLoaded(model)
    assert(string(get_param(model,"SimulationStatus"))=="stopped" && string(get_param(model,"Dirty"))=="off", ...
        "ARES:BarrierModelHasEdits","Stop the experiment and preserve any model edits before preparing it again.");
    close_system(model,0);
end
if isfile(modelPath)
    assert(isfile(metadataPath),"ARES:UnknownBarrierCopy","Existing model has no preparation record: %s",modelPath);
    previous=jsondecode(fileread(metadataPath));
    assert(ares.inference.fileSha256(modelPath)==string(previous.preparedModelSha256), ...
        "ARES:BarrierCopyHasEdits","The saved experiment copy was edited; preserve/move it before regenerating.");
end
baseline=fullfile(projectRoot,"models","ares_single_uav_top.slx");
baselineHash=ares.inference.fileSha256(baseline);
assert(baselineHash==string(deployment.physical_calibration.model_sha256), ...
    "ARES:BarrierPlantChanged","The baseline model differs from the plant used for training calibration.");
report=validateSaitsBarrierMatlab(projectRoot,deploymentPath,pythonExecutable);
ares.inference.prepareSaitsSimulation(projectRoot,true);
% Build elsewhere; a failed compilation must not replace a valid prepared
% file or strand an unverified model at the public path.
buildingFolder=fullfile(folder,"building_"+string(java.util.UUID.randomUUID()));
mkdir(buildingFolder);
buildingPath=fullfile(buildingFolder,model+".slx");
copyfile(baseline,buildingPath); load_system(buildingPath);
try
    wiring=ares.inference.configureSaitsBarrierModel(model,projectRoot,deploymentPath,pythonExecutable,cfg,continueAfterFailure);
    set_param(model,"StopTime",num2str(cfg.simulation.stopTime),"SimulationMode","normal");
    visualization=ares.inference.configureSaitsVisualization(model,projectRoot,showSimulation);
    % Callbacks recreate the original seed-513 schedule on every toolbar Run.
    rootQuoted="'"+replace(projectRoot,"'","''")+"'";
    set_param(model,"InitFcn","ares.inference.prepareSaitsSimulation("+rootQuoted+",true);", ...
        "StopFcn","ares.inference.prepareSaitsSimulation("+rootQuoted+",false);");
    set_param(model,"SimulationCommand","update"); % Compile/check only. Never sim().
    save_system(model,modelPath);
catch exception
    % Preserve the failed copy for diagnosis, but it must not look validated.
    if bdIsLoaded(model), close_system(model,0); end
    rethrow(exception);
end
assert(ares.inference.fileSha256(baseline)==baselineHash,"ARES:BaselineChanged","Baseline model changed.");
prepared.schemaVersion="ares-saits-barrier-preparation-v1";
prepared.projectRoot=projectRoot;
prepared.modelName=model; prepared.modelPath=modelPath;
prepared.preparedModelSha256=ares.inference.fileSha256(modelPath);
prepared.baselineModelSha256=baselineHash;
prepared.deploymentPath=deploymentPath;
prepared.deploymentManifestSha256=report.deploymentManifestSha256;
prepared.selectedCheckpointSha256=string(deployment.provenance.source_checkpoint_sha256);
prepared.pythonExecutable=pythonExecutable;
prepared.numericalValidation=report; prepared.wiring=wiring; prepared.visualization=visualization;
prepared.plantSimulationRun=false;
prepared.continueAfterBarrierFailure=continueAfterFailure;
prepared.telemetrySubdirectory="saits_barrier_intermittent";
prepared.runtimeSources=struct();
files=["scripts/saits_barrier_simulink.py","src/+ares/+inference/SaitsBarrierController.m", ...
    "src/+ares/+inference/configureSaitsBarrierModel.m","src/+ares/+inference/loadSaitsBarrierRuntime.m"];
for k=1:numel(files)
    prepared.runtimeSources.(matlab.lang.makeValidName(files(k)))=ares.inference.fileSha256(fullfile(projectRoot,files(k)));
end
writeJson(metadataPath,prepared);
assignin("base","saitsBarrierPrepared",prepared);
fprintf("Prepared and compiled %s. Simulation has NOT been started.\n",model);
fprintf("When ready, run scripts/captureSaitsIntermittentLidarSingleUav.m.\n");
end

function writeJson(path,value)
fid=fopen(path,"w","n","UTF-8"); assert(fid~=-1,"ARES:WriteFailed","Cannot write %s",path);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
