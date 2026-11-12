function [manifest,telemetry,manifestPath,telemetryPath,summary] = captureCnnSingleEpisode(projectRoot,artifactPath,showSimulation,episodeId)
%CAPTURECNNSINGLEEPISODE Reuse the original intermittent capture and VFH/PID framework.
arguments
    projectRoot (1,1) string
    artifactPath (1,1) string
    showSimulation (1,1) logical
    episodeId (1,1) string
end
assert(isscalar(artifactPath) && isfile(artifactPath),"ARES:MissingCnnArtifact", ...
    "Export a trained CNN artifact first. Missing model.mat: %s",artifactPath);
modelName = "ares_single_uav_cnn_"+erase(episodeId,"-");
baselineName = "ares_single_uav_top";
if bdIsLoaded(modelName) && string(get_param(modelName,"SimulationStatus"))=="stopped"
    stateBlock = modelName+"/ARES CNN Reconstruction";
    previous = struct();
    if getSimulinkBlockHandle(stateBlock)~=-1
        previous = get_param(stateBlock,"UserData");
    end
    generatedComplete = isstruct(previous) && isfield(previous,"aresCnnCaptureComplete") && ...
        previous.aresCnnCaptureComplete;
    cleanFactoryCopy = getSimulinkBlockHandle(stateBlock)==-1 && ...
        string(get_param(modelName,"FileName"))== ...
        fullfile(projectRoot,"results","cnn_mlp_integration",episodeId,modelName+".slx");
    if (generatedComplete || cleanFactoryCopy) && string(get_param(modelName,"Dirty"))=="off"
        close_system(modelName,0);
    end
end
assert(~bdIsLoaded(modelName),"ARES:CnnModelAlreadyLoaded", ...
    "Close the temporary CNN model before starting this capture; it will not be overwritten.");
if bdIsLoaded(baselineName)
    assert(string(get_param(baselineName,"SimulationStatus"))=="stopped", ...
        "ARES:BaselineSimulationRunning","Stop the baseline simulation before initializing a new experiment.");
end
initializer = fullfile(projectRoot,"scripts","initializeSingleUav.m");
baselinePath = fullfile(projectRoot,"models",baselineName+".slx");
copyFolder = fullfile(projectRoot,"results","cnn_mlp_integration",episodeId);
copyPath = fullfile(copyFolder,modelName+".slx");
baselineSha = fileHash(baselinePath);
if isfile(copyPath)
    assert(fileHash(copyPath)==baselineSha,"ARES:CnnModelCopyChanged", ...
        "The temporary model file differs from the baseline. Preserve or move it before rerunning: %s",copyPath);
else
    if ~isfolder(copyFolder), mkdir(copyFolder); end
    copyfile(baselinePath,copyPath);
end
cleanup = onCleanup(@()restoreNominal(modelName,initializer,showSimulation)); %#ok<NASGU>
[cfg,sourceEpisode] = ares.inference.cnnSingleEpisodeConfig(projectRoot,episodeId);
cfg.sensors.lidar.enabled = true;
cfg.sensors.lidar.mode = "intermittent-cnn-mlp";
[schedule,intervals,faultConfiguration] = ...
    ares.inference.createIntermittentSchedule(double(cfg.simulation.stopTime));
assignin("base","SingleUavConfigOverride",cfg);
assignin("base","SingleUavLidarScheduleOverride",schedule);
evalin("base","clear simOut;");
evalin("base","run('"+replace(initializer,"'","''")+"');");
load_system(copyPath);
wiring = ares.inference.configureCnnMlpModel(modelName,artifactPath);
stateBlock = modelName+"/ARES CNN Reconstruction";
set_param(stateBlock,"UserData",struct("aresCnnCaptureComplete",false), ...
    "UserDataPersistent","off");
set_param(modelName,"StopTime",num2str(cfg.simulation.stopTime));
visualization = ares.inference.configureCnnVisualization(modelName,projectRoot,showSimulation,episodeId);
manifest = ares.data.createRunManifest(projectRoot,modelName, ...
    "cnn-single-episode-intermittent-lidar-single-uav",cfg);
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
manifest.groundTruth.condition = "intermittent-lidar-with-cnn";
manifest.groundTruth.faultType = "intermittent-lidar-unavailable";
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
manifest.inference.modelType = "state-only-MLP-CNN-dense-range-reconstruction";
manifest.inference.sourceEpisodeId = episodeId;
deployment=jsondecode(fileread(fullfile(fileparts(artifactPath),"export_manifest.json")));
assert(string(deployment.episode_id)==episodeId,"ARES:WrongCnnEpisode","Artifact belongs to another episode.");
assert(string(deployment.source_manifest_sha256)==fileHash(fullfile(projectRoot,"data","raw",episodeId,"manifest.json")), ...
    "ARES:CnnSourceChanged","Source episode manifest changed after training.");
manifest.inference.trainingRunDirectory=string(deployment.training_run_dir);
manifest.inference.sourceCheckpointSha256=string(deployment.checkpoint_sha256);
assert(isequal(double(sourceEpisode.static.obstacle_centers_enu_m),manifest.static.obstacle_centers_enu_m), ...
    "ARES:CnnSceneChanged","Current scene differs from the trained episode scene.");
manifest.inference.artifactPath = artifactPath;
manifest.inference.artifactSha256 = fileHash(artifactPath);
validationPath = fullfile(fileparts(artifactPath),"matlab_validation.json");
assert(isfile(validationPath),"ARES:MissingCnnValidation","Run validateCnnMlp before capture.");
if isfile(validationPath)
    validation = jsondecode(fileread(validationPath));
    assert(string(validation.modelSha256)==manifest.inference.artifactSha256 && ...
        validation.allParityChecksPassed,"ARES:StaleCnnValidation", ...
        "The model artifact does not match its MATLAB validation report.");
    manifest.inference.numericalValidation = validation;
end
manifest.inference.inputFeatureCount = 12;
manifest.inference.targetFeatureCount = 10740;
manifest.inference.loggedSectorSummaryCount = 14;
manifest.inference.densePredictionAvailableSignal = "cnn_reconstruction_used";
manifest.inference.unusedDensePredictionFillValue = 0;
manifest.inference.historyLength = 0;
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
fprintf("\nCNN intermittent LiDAR run: %s\n",manifest.runId);
disp(array2table(intervals,"VariableNames",["StartTime_s","EndTime_s"]));
quotedRoot="'"+replace(projectRoot,"'","''")+"'";
quotedEpisode="'"+replace(episodeId,"'","''")+"'";
set_param(modelName,"InitFcn","ares.inference.prepareCnnSingleEpisodeSimulation("+quotedRoot+","+quotedEpisode+",true);", ...
    "StopFcn","ares.inference.prepareCnnSingleEpisodeSimulation("+quotedRoot+","+quotedEpisode+",false);");
wallStart = tic;
simOut = sim(modelName);
manifest.runtime.wallSeconds = toc(wallStart);
manifest.runtime.completedAtUtc = utcNow();
telemetry = ares.data.extractRawTelemetry(simOut,manifest);
telemetry = appendLoggedSignals(telemetry,simOut.logsout);
oracle = simOut.logsout.getElement("nominal_lidar_points_xyz_m").Values;
telemetry.oracleLidar.time_s = double(oracle.Time(:));
telemetry.oracleLidar.pointsXyz_m = oracle.Data;
telemetry.oracleLidar.dimensionOrder = "elevation,azimuth,xyz,scan";
telemetry.oracleLidar.frame = "sensor-FLU";
telemetry.oracleLidar.usage = "scoring-only during outages; excluded from CNN inputs";
telemetry.mission.finalGoalPositionNed_m = cfg.mission.waypointsNed(end,:);
telemetry.mission.waypointsNed_m = cfg.mission.waypointsNed;
metrics = ares.data.evaluateAxisRun(telemetry,cfg, ...
    manifest.static.obstacle_centers_ned_m,manifest.static.obstacle_width_m,manifest.static.obstacle_height_m);
assert(metrics.telemetryValid && metrics.initialPositionMatchesPlan,"ARES:InvalidCnnRunState", ...
    "The saved native trajectory is invalid or does not start at the configured position.");
[scan,proof] = makeScanTable(telemetry,schedule,manifest.runId);
used=logical(scan.CnnReconstructionUsed);
if any(used)
    signal=telemetry.signals.cnn_predicted_ranges_m;
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
    manifest.outcome.failureIdentifier = "ARES:CnnRuntimeProofFailed";
    manifest.outcome.failureMessage = strjoin(proof.failedChecks,"; ");
end
telemetry.outcome = manifest.outcome;
telemetry.inferenceMetadata = manifest.inference;
telemetry.lidarFaultMetadata = manifest.groundTruth;
telemetry.lidarFaultMetadata.disabledScanCount = proof.disabledLidarScanCount;
telemetry.lidarFaultMetadata.enabledScanCount = proof.enabledLidarScanCount;
telemetry.lidarFaultMetadata.fillValueM = 7;
telemetry.lidarFaultMetadata.missingnessType = "complete-lidar-scan-outage";
summary = struct("schemaVersion","cnn-single-episode-intermittent-run-summary-v1", ...
    "runId",manifest.runId,"createdAtUtc",manifest.createdAtUtc, ...
    "artifactSha256",manifest.inference.artifactSha256, ...
    "faultSeed",faultConfiguration.randomSeed,"outageIntervalsS",intervals, ...
    "metrics",metrics,"runtimeProof",proof, ...
    "safetyChecksPassed",metrics.safetyChecksPassed,"missionSuccess",metrics.missionSuccess, ...
    "collisionOccurred",metrics.collisionOccurred,"wallSeconds",manifest.runtime.wallSeconds, ...
    "collisionDetectionMethod",metrics.clearanceEvaluation, ...
    "interpretation","Online single-episode CNN reconstruction test; measured LiDAR is used when available, state-only dense prediction during outages.");
outputSubdirectory = "cnn_single_episode_intermittent";
manifestPath = ares.data.saveRuntimeManifest(projectRoot,manifest,outputSubdirectory);
telemetryPath = ares.data.saveRawTelemetry(projectRoot,telemetry,outputSubdirectory);
runFolder = string(fileparts(telemetryPath));
scanPath = fullfile(runFolder,"scans.csv");
summaryPath = fullfile(runFolder,"summary.json");
assert(~isfile(scanPath) && ~isfile(summaryPath),"ARES:CnnArtifactAlreadyExists", ...
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
fprintf("CNN run folder: %s\n",runFolder);
fprintf("Observed outages: %d; reconstruction-used scans: %d; controller-influenced scans: %d\n", ...
    proof.observedOutageCount,proof.reconstructionUsedScanCount,proof.controllerInfluencedScanCount);
fprintf("Mission success: %d; safety checks passed: %d; sampled collision: %d\n", ...
    metrics.missionSuccess,metrics.safetyChecksPassed,metrics.collisionOccurred);
assert(proof.instrumentationChecksPassed,"ARES:CnnRuntimeProofFailed", ...
    "Run artifacts were saved, but online instrumentation checks failed: %s", ...
    strjoin(proof.failedChecks,"; "));
set_param(stateBlock,"UserData",struct("aresCnnCaptureComplete",true));
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
if any(name==["cnn_predicted_sectors","cnn_controller_sectors"]) && width==14
    unit = "mixed"; components = sectorComponents;
    componentUnits = [repmat("m",1,7),repmat("fraction",1,7)];
elseif any(name==["cnn_input_features","cnn_input_observed_mask"]) && width==12
    unit="mixed"; components=["north","east","down","vn","ve","vd", ...
        "sin_roll","cos_roll","sin_pitch","cos_pitch","sin_yaw","cos_yaw"];
    componentUnits=[repmat("m",1,3),repmat("m/s",1,3),repmat("dimensionless",1,6)];
    if name=="cnn_input_observed_mask"
        unit="boolean"; componentUnits=repmat("boolean",1,12);
    end
elseif name=="cnn_predicted_ranges_m" && width==10740
    unit="m"; components=compose("beam_%05d",1:10740); componentUnits=repmat("m",1,10740);
elseif width==3 && endsWith(name,"_ned_m")
    unit = "m"; components = ["north","east","down"];
    componentUnits = repmat(unit,1,3);
elseif name=="cnn_inference_seconds"
    unit = "s"; componentUnits = unit;
elseif name=="cnn_fallback_reason"
    unit = "code"; componentUnits = unit;
elseif width==1 && startsWith(name,"cnn_")
    unit = "boolean"; componentUnits = unit;
end
end

function [scan,proof] = makeScanTable(telemetry,schedule,runId)
signals = telemetry.signals;
time = double(signals.nominal_observation.time_s(:));
n = numel(time);
assert(n>=2 && all(diff(time)>0) && max(abs(diff(time)-0.1))<1e-6, ...
    "ARES:InvalidCnnScanTimeline","Expected an uninterrupted 10 Hz scan timeline.");
required = ["cnn_reconstruction_used","cnn_model_ready","cnn_prediction_valid", ...
    "cnn_fallback_reason","cnn_inference_seconds","cnn_predicted_sectors", ...
    "cnn_controller_sectors","cnn_input_features","cnn_input_observed_mask", ...
    "cnn_guidance_enabled","cnn_measured_available","cnn_backend_native", ...
    "cnn_selected_reference_ned_m","cnn_waypoint_reference_ned_m","cnn_reference_delta_ned_m"];
for name = required
    assert(isfield(signals,name),"ARES:MissingCnnProofSignal", ...
        "Missing online proof signal: %s",name);
end
available = interp1(double(schedule.Time(:)),double(schedule.Data(:)),time,"previous")>=0.5;
used = sample(signals.cnn_reconstruction_used,time,1)>=0.5;
ready = sample(signals.cnn_model_ready,time,1)>=0.5;
validPrediction = sample(signals.cnn_prediction_valid,time,1)>=0.5;
measuredAvailable = sample(signals.cnn_measured_available,time,1)>=0.5;
backendNative = sample(signals.cnn_backend_native,time,1)>=0.5;
guidance = sample(signals.cnn_guidance_enabled,time,1)>=0.5;
reason = sample(signals.cnn_fallback_reason,time,1);
inferenceSeconds = sample(signals.cnn_inference_seconds,time,1);
rawPrediction = sample(signals.cnn_predicted_sectors,time,14);
controllerSectors = sample(signals.cnn_controller_sectors,time,14);
input = sample(signals.cnn_input_features,time,12);
observedMask = sample(signals.cnn_input_observed_mask,time,12);
selectedReference = sample(signals.cnn_selected_reference_ned_m,time,3);
waypointReference = sample(signals.cnn_waypoint_reference_ned_m,time,3);
referenceDelta = sample(signals.cnn_reference_delta_ned_m,time,3);
position = sample(signals.position_ned_m,time,3);
velocity = sample(signals.velocity_ned_mps,time,3);
oracleTargets = oracleSectorTargets(signals.nominal_observation.values, ...
    logical(signals.nominal_lidar_valid_return_mask.values));
dropout = [false;available(1:end-1) & ~available(2:end)];
recovery = [false;~available(1:end-1) & available(2:end)];
age = zeros(n,1); lastHealthy = time(1);
for index = 1:n
    if available(index), lastHealthy = time(index); end
    age(index) = time(index)-lastHealthy;
end
normDelta = vecnorm(referenceDelta,2,2);
influenced = used & ~available & normDelta>1e-5;
proof.scanCount = n;
proof.observedOutageCount = nnz(dropout);
proof.observedRecoveryCount = nnz(recovery);
proof.disabledLidarScanCount = nnz(~available);
proof.enabledLidarScanCount = nnz(available);
proof.reconstructionUsedScanCount = nnz(used & ~available);
proof.nativeInferenceScanCount = nnz(used & ~available & backendNative);
proof.nativeBackendSelected = all(backendNative);
proof.warmupFallbackScanCount = nnz(~available & reason==1);
proof.invalidPredictionFallbackScanCount = nnz(~available & reason==2);
proof.controllerInfluencedScanCount = nnz(influenced);
proof.controllerInfluencedDuringOutage = any(influenced);
proof.maximumUsedReferenceDifferenceM = 0;
if any(used & ~available)
    proof.maximumUsedReferenceDifferenceM = max(normDelta(used & ~available));
end
proof.totalInferenceSeconds = sum(inferenceSeconds);
proof.meanInferenceMilliseconds = 0;
proof.maximumInferenceMilliseconds = 0;
if any(used)
    proof.meanInferenceMilliseconds = 1000*mean(inferenceSeconds(used));
    proof.maximumInferenceMilliseconds = 1000*max(inferenceSeconds(used));
end
proof.oracleOutageScansWithReturns = nnz(~available & any(oracleTargets(:,8:14)>0,2));
proof.failedChecks = strings(0,1);
checks = [available(1) && available(end),nnz(dropout)==3 && nnz(recovery)==3, ...
    isequal(measuredAvailable,available),all(~used | (~available & ready & validPrediction)), ...
    any(used & ~available),all(guidance==(available | used)), ...
    max(abs(referenceDelta-(selectedReference-waypointReference)),[],"all")<=1e-4, ...
    all(observedMask==0 | observedMask==1,"all"), ...
    all(observedMask==1,"all"), ...
    all(inferenceSeconds>=0)];
labels = ["LiDAR must start and end healthy","All three outages and recoveries must be observed", ...
    "Measured availability must equal the original schedule", ...
    "Reconstruction use requires outage, loaded model and valid prediction", ...
    "At least one outage scan must use reconstruction", ...
    "Guidance enable must equal measured availability OR reconstruction use", ...
    "Reference delta must equal selected minus waypoint reference", ...
    "CNN observed mask must be binary","All CNN channels must be measured motion features; no LiDAR inputs", ...
    "Inference timings must be nonnegative"];
proof.failedChecks = labels(~checks);
proof.instrumentationChecksPassed = all(checks);
scan = table(repmat(runId,n,1),uint32((1:n)'),time, ...
    'VariableNames',{'RunId','SampleIndex','TimeS'});
scan.LidarMeasuredAvailable = available;
scan.LidarDropoutEvent = dropout; scan.LidarRecoveryEvent = recovery;
scan.OutageAgeS = age; scan.CnnReconstructionUsed = used;
scan.CnnModelReady = ready; scan.CnnPredictionValid = validPrediction;
scan.CnnFallbackReason = reason; scan.CnnGuidanceEnabled = guidance;
scan.CnnInferenceSeconds = inferenceSeconds;
scan.CnnBackendNative = backendNative;
scan.ControllerReferenceDifferenceM = normDelta;
scan.ControllerInfluencedDuringOutage = influenced;
for axis = 1:3
    suffix = ["North","East","Down"]; name = suffix(axis);
    scan.("Position"+name+"M") = position(:,axis);
    scan.("Velocity"+name+"Mps") = velocity(:,axis);
    scan.("SelectedReference"+name+"M") = selectedReference(:,axis);
    scan.("WaypointReference"+name+"M") = waypointReference(:,axis);
    scan.("ReferenceDelta"+name+"M") = referenceDelta(:,axis);
end
actionNames = ["command_roll_rad","command_pitch_rad","command_yaw_rate_radps","command_thrust_n"];
for index = 1:numel(actionNames)
    scan.(actionNames(index)) = sample(signals.(actionNames(index)),time,1);
end
for index = 1:14
    scan.(sprintf("CnnPredictedSector_%02d",index)) = rawPrediction(:,index);
    scan.(sprintf("ControllerSector_%02d",index)) = controllerSectors(:,index);
    scan.(sprintf("OracleScoringTarget_%02d",index)) = oracleTargets(:,index);
end
for index = 1:12
    scan.(sprintf("CnnInput_%02d",index)) = input(:,index);
    scan.(sprintf("CnnInputObserved_%02d",index)) = logical(observedMask(:,index));
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

function targets = oracleSectorTargets(ranges,valid)
n = size(ranges,1);
gridRanges = reshape(double(ranges.'),15,716,n);
gridValid = reshape(logical(valid.'),15,716,n);
bearing = -linspace(-179,179,716);
edges = linspace(-60,60,8);
targets = zeros(n,14);
for sector = 1:7
    columns = bearing>=edges(sector) & bearing<edges(sector+1);
    if sector==7, columns = bearing>=edges(sector) & bearing<=edges(sector+1); end
    hits = reshape(gridValid(:,columns,:),[],n)';
    selected = reshape(gridRanges(:,columns,:),[],n)';
    selected(~hits) = inf;
    nearest = min(selected,[],2); nearest(~isfinite(nearest)) = 7;
    targets(:,sector) = nearest;
    targets(:,sector+7) = mean(hits,2);
end
end

function values = sample(signal,query,width)
time = double(signal.time_s(:));
[time,last] = unique(time,"last");
native = double(signal.values(last,:));
assert(size(native,2)==width,"ARES:UnexpectedCnnSignalWidth","Unexpected proof-signal width.");
tolerance = 1e-9;
assert(query(1)>=time(1)-tolerance && query(end)<=time(end)+tolerance, ...
    "ARES:CnnSignalTimeCoverage","A proof signal does not cover the scan timeline.");
query = min(max(query,time(1)),time(end));
values = interp1(time,native,query,"previous");
assert(all(isfinite(values),"all"),"ARES:NonfiniteCnnProofSignal","Proof signals must be finite.");
end

function restoreNominal(modelName,initializer,showSimulation)
if bdIsLoaded(modelName)
    stateBlock = modelName+"/ARES CNN Reconstruction";
    status = struct();
    if getSimulinkBlockHandle(stateBlock)~=-1
        status = get_param(stateBlock,"UserData");
    end
    keepOpen = showSimulation && isstruct(status) && ...
        isfield(status,"aresCnnCaptureComplete") && status.aresCnnCaptureComplete;
    if ~keepOpen
        try, close_system(modelName,0); catch failure
            warning("ARES:CnnModelCleanupFailed","Could not close temporary model: %s",failure.message);
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
