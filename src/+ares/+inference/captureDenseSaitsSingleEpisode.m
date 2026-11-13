function [manifest,telemetry,manifestPath,telemetryPath,summary] = captureDenseSaitsSingleEpisode(projectRoot,artifactPath,showSimulation,episodeId,lidarMode)
%CAPTUREDENSESAITSSINGLEEPISODE Capture intermittent or fully disabled LiDAR.
arguments
    projectRoot (1,1) string
    artifactPath (1,1) string
    showSimulation (1,1) logical
    episodeId (1,1) string
    lidarMode (1,1) string {mustBeMember(lidarMode,["intermittent","disabled"])} = "intermittent"
end
assert(isscalar(artifactPath) && isfile(artifactPath),"ARES:MissingDenseSaitsArtifact", ...
    "Export a trained DENSE SAITS artifact first. Missing model.mat: %s",artifactPath);
fullyDisabled=lidarMode=="disabled";
modelPrefix="ares_single_uav_saits_dense_";
integrationFolder="saits_dense_integration";
scenarioName="saits-dense-single-episode-intermittent-lidar-single-uav";
outputSubdirectory="saits_dense_single_episode_intermittent";
condition="intermittent-lidar-with-dense-saits";
summarySchema="saits-dense-single-episode-intermittent-run-summary-v1";
if fullyDisabled
    modelPrefix="ares_uav_saits_dense_off_";
    integrationFolder="saits_dense_disabled_integration";
    scenarioName="saits-dense-single-episode-lidar-disabled-single-uav";
    outputSubdirectory="saits_dense_single_episode_disabled";
    condition="lidar-disabled-with-dense-saits";
    summarySchema="saits-dense-single-episode-disabled-run-summary-v1";
end
modelName = modelPrefix+erase(episodeId,"-");
baselineName = "ares_single_uav_top";
if bdIsLoaded(modelName) && string(get_param(modelName,"SimulationStatus"))=="stopped"
    stateBlock = modelName+"/ARES DENSE SAITS Reconstruction";
    previous = struct();
    if getSimulinkBlockHandle(stateBlock)~=-1
        previous = get_param(stateBlock,"UserData");
    end
    generatedComplete = isstruct(previous) && isfield(previous,"aresDenseSaitsCaptureComplete") && ...
        previous.aresDenseSaitsCaptureComplete;
    cleanFactoryCopy = getSimulinkBlockHandle(stateBlock)==-1 && ...
        string(get_param(modelName,"FileName"))== ...
        fullfile(projectRoot,"results",integrationFolder,episodeId,modelName+".slx");
    if (generatedComplete || cleanFactoryCopy) && string(get_param(modelName,"Dirty"))=="off"
        close_system(modelName,0);
    end
end
assert(~bdIsLoaded(modelName),"ARES:DenseSaitsModelAlreadyLoaded", ...
    "Close the temporary DENSE SAITS model before starting this capture; it will not be overwritten.");
if bdIsLoaded(baselineName)
    assert(string(get_param(baselineName,"SimulationStatus"))=="stopped", ...
        "ARES:BaselineSimulationRunning","Stop the baseline simulation before initializing a new experiment.");
end
initializer = fullfile(projectRoot,"scripts","initializeSingleUav.m");
baselinePath = fullfile(projectRoot,"models",baselineName+".slx");
copyFolder = fullfile(projectRoot,"results",integrationFolder,episodeId);
copyPath = fullfile(copyFolder,modelName+".slx");
baselineSha = fileHash(baselinePath);
if isfile(copyPath)
    assert(fileHash(copyPath)==baselineSha,"ARES:DenseSaitsModelCopyChanged", ...
        "The temporary model file differs from the baseline. Preserve or move it before rerunning: %s",copyPath);
else
    if ~isfolder(copyFolder), mkdir(copyFolder); end
    copyfile(baselinePath,copyPath);
end
cleanup = onCleanup(@()restoreNominal(modelName,initializer,showSimulation)); %#ok<NASGU>
[cfg,sourceEpisode] = ares.inference.denseSaitsSingleEpisodeConfig(projectRoot,episodeId);
cfg.sensors.lidar.enabled = ~fullyDisabled;
cfg.sensors.lidar.mode = lidarMode+"-dense-saits";
[schedule,intervals,faultConfiguration] = ...
    ares.inference.denseSaitsLidarScenario(double(cfg.simulation.stopTime),lidarMode);
assignin("base","SingleUavConfigOverride",cfg);
assignin("base","SingleUavLidarScheduleOverride",schedule);
evalin("base","clear simOut;");
evalin("base","run('"+replace(initializer,"'","''")+"');");
load_system(copyPath);
wiring = ares.inference.configureDenseSaitsModel(modelName,artifactPath);
stateBlock = modelName+"/ARES DENSE SAITS Reconstruction";
set_param(stateBlock,"UserData",struct("aresDenseSaitsCaptureComplete",false), ...
    "UserDataPersistent","off");
set_param(modelName,"StopTime",num2str(cfg.simulation.stopTime));
visualization = ares.inference.configureDenseSaitsVisualization(modelName,projectRoot,showSimulation,episodeId,lidarMode);
manifest = ares.data.createRunManifest(projectRoot,modelName,scenarioName,cfg);
manifest.static.maximum_sensor_range_m = 7;
manifest.static.nominal_observation_dimension = 10740;
manifest.static.lidar_grid_size = [15 716];
manifest.static.observation_channel_order = "elevation-index-fastest-then-azimuth-index";
manifest.static.obstacle_centers_enu_m = evalin("base","ObstaclePositions");
manifest.static.obstacle_centers_ned_m = manifest.static.obstacle_centers_enu_m(:,[2 1]);
manifest.static.obstacle_width_m = double(evalin("base","ObstacleWidth"));
manifest.static.obstacle_height_m = double(evalin("base","ObstacleHeight"));
manifest.static.obstacle_mesh_reference_frame = "ENU";
manifest.static.oracle_usage = "Nominal raw LiDAR retained for scoring only during outages; never a reconstruction input.";
manifest.groundTruth.condition = condition;
manifest.groundTruth.faultType = "intermittent-lidar-unavailable";
if fullyDisabled, manifest.groundTruth.faultType="lidar-unavailable"; end
manifest.groundTruth.faultActive = true;
manifest.groundTruth.faultStartTime_s = intervals(1,1);
manifest.groundTruth.faultEndTime_s = intervals(end,2);
manifest.groundTruth.outageIntervals_s = intervals;
manifest.groundTruth.randomSeed = faultConfiguration.randomSeed;
manifest.groundTruth.numberOfOutages = faultConfiguration.numberOfOutages;
manifest.groundTruth.faultWindowEndTime_s = faultConfiguration.faultWindowEndTime_s;
manifest.groundTruth.scheduleTime_s = double(schedule.Time(:));
manifest.groundTruth.scheduleEnabled = logical(schedule.Data(:));
manifest.groundTruth.faultConfiguration = faultConfiguration;
manifest.inference.modelType = "history-and-motion-SAITS-dense-range-reconstruction";
manifest.inference.sourceEpisodeId = episodeId;
deployment=jsondecode(fileread(fullfile(fileparts(artifactPath),"export_manifest.json")));
assert(string(deployment.episode_id)==episodeId,"ARES:WrongDenseSaitsEpisode","Artifact belongs to another episode.");
assert(string(deployment.source_manifest_sha256)==fileHash(fullfile(projectRoot,"data","raw",episodeId,"manifest.json")), ...
    "ARES:DenseSaitsSourceChanged","Source episode manifest changed after training.");
manifest.inference.trainingRunDirectory=string(deployment.training_run_dir);
manifest.inference.sourceCheckpointSha256=string(deployment.checkpoint_sha256);
assert(isequal(double(sourceEpisode.static.obstacle_centers_enu_m),manifest.static.obstacle_centers_enu_m), ...
    "ARES:DenseSaitsSceneChanged","Current scene differs from the trained episode scene.");
manifest.inference.artifactPath = artifactPath;
manifest.inference.artifactSha256 = fileHash(artifactPath);
validationPath = fullfile(fileparts(artifactPath),"matlab_validation.json");
assert(isfile(validationPath),"ARES:MissingDenseSaitsValidation","Run validateDenseSaits before capture.");
if isfile(validationPath)
    validation = jsondecode(fileread(validationPath));
    assert(string(validation.modelSha256)==manifest.inference.artifactSha256 && ...
        validation.allParityChecksPassed,"ARES:StaleDenseSaitsValidation", ...
        "The model artifact does not match its MATLAB validation report.");
    manifest.inference.numericalValidation = validation;
end
manifest.inference.inputFeatureCount = 10752;
manifest.inference.targetFeatureCount = 10740;
manifest.inference.loggedSectorSummaryCount = 14;
manifest.inference.densePredictionAvailableSignal = "saits_dense_reconstruction_used";
manifest.inference.unusedDensePredictionFillValue = 0;
exportedModel=load(artifactPath,"model");
manifest.inference.historyLength = double(exportedModel.model.nSteps);
manifest.inference.trainingStage = "reconstruction-only";
manifest.inference.barrierEnabled = false;
manifest.inference.lidarMode = lidarMode;
manifest.inference.coldStartWithoutMeasuredLidar = fullyDisabled;
manifest.inference.historyInitialization = "empty zero-valued buffer with false observation masks; no recorded or predicted scan treated as measured";
manifest.inference.scanPeriodS = 0.1;
manifest.inference.wiring = wiring;
manifest.runtime.visualization = visualization;
manifest.inference.sectorBodyBearingEdgesDeg = linspace(-60,60,8);
manifest.inference.controllerAdapter = "full predicted 15x716 range grid projected onto recorded sensor rays; ranges clipped to [0,7]m";
manifest.inference.baselineModelPath = baselinePath;
manifest.inference.baselineModelSha256 = baselineSha;
manifest.inference.temporaryModelPath = copyPath;
manifest.inference.fallbackReasonCodes = struct("healthy",0,"unusedWarmupCode",1, ...
    "invalidPrediction",2,"modelUsed",3);
fprintf("\nDENSE SAITS %s LiDAR run: %s\n",lidarMode,manifest.runId);
disp(array2table(intervals,"VariableNames",["StartTime_s","EndTime_s"]));
quotedRoot="'"+replace(projectRoot,"'","''")+"'";
quotedEpisode="'"+replace(episodeId,"'","''")+"'";
quotedMode="'"+lidarMode+"'";
set_param(modelName,"InitFcn","ares.inference.prepareDenseSaitsSingleEpisodeSimulation("+quotedRoot+","+quotedEpisode+",true,"+quotedMode+");", ...
    "StopFcn","ares.inference.prepareDenseSaitsSingleEpisodeSimulation("+quotedRoot+","+quotedEpisode+",false,"+quotedMode+");");
wallStart = tic;
simOut = sim(modelName);
manifest.runtime.wallSeconds = toc(wallStart);
manifest.runtime.completedAtUtc = utcNow();
if fullyDisabled, manifest.groundTruth.faultEndTime_s=double(simOut.tout(end)); end
telemetry = ares.data.extractRawTelemetry(simOut,manifest);
telemetry = appendLoggedSignals(telemetry,simOut.logsout);
oracle = simOut.logsout.getElement("nominal_lidar_points_xyz_m").Values;
telemetry.oracleLidar.time_s = double(oracle.Time(:));
telemetry.oracleLidar.pointsXyz_m = oracle.Data;
telemetry.oracleLidar.dimensionOrder = "elevation,azimuth,xyz,scan";
telemetry.oracleLidar.frame = "sensor-FLU";
telemetry.oracleLidar.usage = "scoring-only during outages; excluded from DENSE SAITS inputs";
telemetry.mission.finalGoalPositionNed_m = cfg.mission.waypointsNed(end,:);
telemetry.mission.waypointsNed_m = cfg.mission.waypointsNed;
metrics = ares.data.evaluateAxisRun(telemetry,cfg, ...
    manifest.static.obstacle_centers_ned_m,manifest.static.obstacle_width_m,manifest.static.obstacle_height_m);
assert(metrics.telemetryValid && metrics.initialPositionMatchesPlan,"ARES:InvalidDenseSaitsRunState", ...
    "The saved native trajectory is invalid or does not start at the configured position.");
[scan,proof] = ares.inference.denseSaitsScanTable(telemetry,schedule,manifest.runId,lidarMode);
used=logical(scan.DenseSaitsReconstructionUsed);
if any(used)
    signal=telemetry.signals.saits_dense_predicted_ranges_m;
    [nativeTime,last]=unique(double(signal.time_s(:)),"last");
    predicted=interp1(nativeTime,double(signal.values(last,:)),scan.TimeS,"previous");
    error=predicted(used,:)-double(telemetry.signals.nominal_observation.values(used,:));
    proof.rawDenseOutageRmseM=sqrt(mean(error.^2,"all"));
    proof.rawDenseOutageMaeM=mean(abs(error),"all");
end
telemetry = addFaultTelemetry(telemetry,scan);
telemetry.metrics = metrics;
manifest.metrics = metrics;
manifest.inference.runtimeProof = proof;
manifest.outcome.safetyChecksPassed = metrics.safetyChecksPassed;
manifest.outcome.missionSuccess = metrics.missionSuccess;
manifest.outcome.collisionOccurred = metrics.collisionOccurred;
manifest.outcome.terminationReason = metrics.terminationReason;
manifest.outcome.observedOutageCount = proof.observedOutageCount;
manifest.outcome.observedRecoveryCount = proof.observedRecoveryCount;
manifest.outcome.reconstructionUsedScanCount = proof.reconstructionUsedScanCount;
manifest.outcome.controllerInfluencedDuringOutage = proof.controllerInfluencedDuringOutage;
manifest.outcome.instrumentationChecksPassed = proof.instrumentationChecksPassed;
manifest.outcome.failureIdentifier = "";
manifest.outcome.failureMessage = "";
if ~proof.instrumentationChecksPassed
    manifest.outcome.failureIdentifier = "ARES:DenseSaitsRuntimeProofFailed";
    manifest.outcome.failureMessage = strjoin(proof.failedChecks,"; ");
end
telemetry.outcome = manifest.outcome;
telemetry.inferenceMetadata = manifest.inference;
telemetry.lidarFaultMetadata = manifest.groundTruth;
telemetry.lidarFaultMetadata.disabledScanCount = proof.disabledLidarScanCount;
telemetry.lidarFaultMetadata.enabledScanCount = proof.enabledLidarScanCount;
telemetry.lidarFaultMetadata.fillValueM = 7;
telemetry.lidarFaultMetadata.missingnessType = "complete-lidar-scan-outage";
summary = struct("schemaVersion",summarySchema, ...
    "runId",manifest.runId,"createdAtUtc",manifest.createdAtUtc, ...
    "artifactSha256",manifest.inference.artifactSha256, ...
    "faultSeed",faultConfiguration.randomSeed,"outageIntervalsS",intervals, ...
    "metrics",metrics,"runtimeProof",proof, ...
    "safetyChecksPassed",metrics.safetyChecksPassed,"missionSuccess",metrics.missionSuccess, ...
    "collisionOccurred",metrics.collisionOccurred,"wallSeconds",manifest.runtime.wallSeconds, ...
    "collisionDetectionMethod",metrics.clearanceEvaluation, ...
    "lidarMode",lidarMode,"coldStartWithoutMeasuredLidar",fullyDisabled, ...
    "interpretation","Online single-episode dense SAITS reconstruction; masked causal LiDAR history and motion predict outage scans; original VFH/PID controller.");
manifestPath = ares.data.saveRuntimeManifest(projectRoot,manifest,outputSubdirectory);
telemetryPath = ares.data.saveRawTelemetry(projectRoot,telemetry,outputSubdirectory);
runFolder = string(fileparts(telemetryPath));
scanPath = fullfile(runFolder,"scans.csv");
summaryPath = fullfile(runFolder,"summary.json");
assert(~isfile(scanPath) && ~isfile(summaryPath),"ARES:DenseSaitsArtifactAlreadyExists", ...
    "Run summary artifacts already exist and will not be overwritten.");
writetable(scan,scanPath);
writeJson(summaryPath,summary);
savedManifest = jsondecode(fileread(manifestPath));
saved = load(telemetryPath,"telemetry");
assert(string(savedManifest.runId)==manifest.runId && ...
    string(saved.telemetry.runId)==manifest.runId,"ARES:RunIdMismatch", ...
    "Saved manifest and telemetry IDs differ.");
assert(fileHash(baselinePath)==baselineSha,"ARES:BaselineModelChanged", ...
    "The saved baseline model changed during the integration run.");
fprintf("DENSE SAITS run folder: %s\n",runFolder);
fprintf("Observed outages: %d; reconstruction-used scans: %d; controller-influenced scans: %d\n", ...
    proof.observedOutageCount,proof.reconstructionUsedScanCount,proof.controllerInfluencedScanCount);
fprintf("Mission success: %d; safety checks passed: %d; sampled collision: %d\n", ...
    metrics.missionSuccess,metrics.safetyChecksPassed,metrics.collisionOccurred);
assert(proof.instrumentationChecksPassed,"ARES:DenseSaitsRuntimeProofFailed", ...
    "Run artifacts were saved, but online instrumentation checks failed: %s", ...
    strjoin(proof.failedChecks,"; "));
set_param(stateBlock,"UserData",struct("aresDenseSaitsCaptureComplete",true));
if showSimulation
    fprintf("Diagram and animation remain open. Run this capture script again to save a new episode.\n");
end
end

function telemetry = appendLoggedSignals(telemetry,logs)
names = string(logs.getElementNames());
skipped = strings(0,1);
for index = 1:numel(names)
    name = names(index);
    if name=="nominal_lidar_points_xyz_m" || isfield(telemetry.signals,name)
        continue;
    end
    element = logs.getElement(index);
    series = element.Values;
    if ~isa(series,"timeseries") || ...
            ~(isnumeric(series.Data) || islogical(series.Data)) || isempty(series.Time)
        skipped(end+1,1) = name; %#ok<AGROW>
        continue;
    end
    width = numel(series.Data)/numel(series.Time);
    assert(width==fix(width) && width>0,"ARES:InvalidAdditionalSignal", ...
        "Logged signal %s cannot be aligned to its timestamps.",name);
    signal = ares.data.timeseriesToSignal(series,double(width));
    signal.name = name;
    [signal.unit,signal.components,signal.componentUnits] = signalMetadata(name,width);
    signal.sourceBlock = "";
    try, signal.sourceBlock = string(element.BlockPath.getBlock(1)); catch, end
    field = string(matlab.lang.makeValidName(name));
    assert(~isfield(telemetry.signals,field),"ARES:AdditionalSignalNameCollision", ...
        "Additional logging field collides after name conversion: %s",name);
    telemetry.signals.(field) = signal;
end
telemetry.additionalLoggingMetadata.skippedNonNumericOrNonTimeseriesSignals = skipped;
end

function [unit,components,componentUnits] = signalMetadata(name,width)
unit = "source-signal-units";
components = compose("channel_%03d",1:width);
componentUnits = repmat(unit,1,width);
sectorComponents = [compose("sector_%02d_distance_m",0:6), ...
    compose("sector_%02d_return_fraction",0:6)];
if any(name==["saits_dense_predicted_sectors","saits_dense_controller_sectors"]) && width==14
    unit = "mixed"; components = sectorComponents;
    componentUnits = [repmat("m",1,7),repmat("fraction",1,7)];
elseif any(name==["saits_dense_input_features","saits_dense_input_observed_mask"]) && width==10752
    unit="mixed"; components=[compose("beam_%05d",1:10740),"north","east","down","vn","ve","vd", ...
        "sin_roll","cos_roll","sin_pitch","cos_pitch","sin_yaw","cos_yaw"];
    componentUnits=[repmat("m",1,10743),repmat("m/s",1,3),repmat("dimensionless",1,6)];
    if name=="saits_dense_input_observed_mask"
        unit="boolean"; componentUnits=repmat("boolean",1,10752);
    end
elseif name=="saits_dense_predicted_ranges_m" && width==10740
    unit="m"; components=compose("beam_%05d",1:10740); componentUnits=repmat("m",1,10740);
elseif width==3 && endsWith(name,"_ned_m")
    unit = "m"; components = ["north","east","down"];
    componentUnits = repmat(unit,1,3);
elseif name=="saits_dense_inference_seconds"
    unit = "s"; componentUnits = unit;
elseif name=="saits_dense_fallback_reason"
    unit = "code"; componentUnits = unit;
elseif width==1 && startsWith(name,"saits_dense_")
    unit = "boolean"; componentUnits = unit;
end
end

function telemetry = addFaultTelemetry(telemetry,scan)
time = scan.TimeS; available = logical(scan.LidarMeasuredAvailable);
width = 10740;
missing = repmat(~available,1,width);
degraded = telemetry.signals.nominal_observation.values;
degraded(missing) = 7;
entries = {"lidar_commanded_on",available;"lidar_failure_active",~available; ...
    "lidar_data_received",available;"lidar_data_valid",available; ...
    "lidar_dropout_event",logical(scan.LidarDropoutEvent); ...
    "latest_lidar_scan_age_s",scan.OutageAgeS;"time_since_last_valid_lidar_s",scan.OutageAgeS; ...
    "sensor_measurement_timestamp_s",time;"degraded_observation",degraded; ...
    "degraded_lidar_ranges_m",degraded;"missing_observation_mask",missing; ...
    "observation_availability_mask",~missing; ...
    "missing_observation_ground_truth",telemetry.signals.nominal_observation.values; ...
    "available_observation_dimension",sum(~missing,2);"missing_observation_dimension",sum(missing,2)};
for index = 1:size(entries,1)
    name = string(entries{index,1}); values = entries{index,2};
    signal = ares.data.timeseriesToSignal(timeseries(values,time),double(size(values,2)));
    signal.name = name; signal.components = "value";
    if size(values,2)==width
        signal.components = telemetry.signals.nominal_observation.components;
        signal.unit = "m";
        if contains(name,"mask"), signal.unit = "boolean"; end
    elseif endsWith(name,"_s")
        signal.unit = "s";
    elseif endsWith(name,"_dimension")
        signal.unit = "count";
    else
        signal.unit = "boolean";
    end
    telemetry.signals.(name) = signal;
end
end

function restoreNominal(modelName,initializer,showSimulation)
if bdIsLoaded(modelName)
    stateBlock = modelName+"/ARES DENSE SAITS Reconstruction";
    status = struct();
    if getSimulinkBlockHandle(stateBlock)~=-1
        status = get_param(stateBlock,"UserData");
    end
    keepOpen = showSimulation && isstruct(status) && ...
        isfield(status,"aresDenseSaitsCaptureComplete") && status.aresDenseSaitsCaptureComplete;
    if ~keepOpen
        try, close_system(modelName,0); catch failure
            warning("ARES:DenseSaitsModelCleanupFailed","Could not close temporary model: %s",failure.message);
        end
    end
end
evalin("base","clear SingleUavConfigOverride SingleUavLidarScheduleOverride;");
try
    evalin("base","run('"+replace(initializer,"'","''")+"');");
catch failure
    warning("ARES:NominalRestorationFailed","Nominal restoration failed: %s",failure.message);
end
end

function hash = fileHash(path)
fid = fopen(path,"rb"); assert(fid~=-1,"ARES:SourceReadFailed","Cannot read: %s",path);
cleanup = onCleanup(@()fclose(fid));
digest = java.security.MessageDigest.getInstance("SHA-256");
while ~feof(fid), digest.update(typecast(fread(fid,1048576,"*uint8"),'int8')); end
hash = lower(string(reshape(dec2hex(typecast(digest.digest(),"uint8"),2).',1,[])));
end

function writeJson(path,value)
assert(~isfile(path),"ARES:ArtifactAlreadyExists","Artifact already exists: %s",path);
fid = fopen(path,"w","n","UTF-8"); assert(fid~=-1,"ARES:WriteFailed","Cannot write: %s",path);
cleanup = onCleanup(@()fclose(fid));
fprintf(fid,"%s",jsonencode(value,PrettyPrint=true));
end

function value = utcNow()
value = string(datetime("now","TimeZone","UTC","Format","yyyy-MM-dd'T'HH:mm:ss.SSSXXX"));
end
