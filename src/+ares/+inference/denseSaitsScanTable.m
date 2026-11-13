function [scan,proof] = denseSaitsScanTable(telemetry,schedule,runId,lidarMode)
%DENSESAITSSCANTABLE Audit actual masked inputs and mode-specific coverage.
arguments
    telemetry (1,1) struct
    schedule (1,1) timeseries
    runId (1,1) string
    lidarMode (1,1) string {mustBeMember(lidarMode,["intermittent","disabled"])} = "intermittent"
end
signals = telemetry.signals;
time = double(signals.nominal_observation.time_s(:));
n = numel(time);
assert(n>=2 && all(diff(time)>0) && max(abs(diff(time)-0.1))<1e-6, ...
    "ARES:InvalidDenseSaitsScanTimeline","Expected an uninterrupted 10 Hz scan timeline.");
required = ["saits_dense_reconstruction_used","saits_dense_model_ready","saits_dense_prediction_valid", ...
    "saits_dense_fallback_reason","saits_dense_inference_seconds","saits_dense_predicted_sectors", ...
    "saits_dense_controller_sectors","saits_dense_input_features","saits_dense_input_observed_mask", ...
    "saits_dense_guidance_enabled","saits_dense_measured_available","saits_dense_backend_native", ...
    "saits_dense_selected_reference_ned_m","saits_dense_waypoint_reference_ned_m","saits_dense_reference_delta_ned_m"];
for name = required
    assert(isfield(signals,name),"ARES:MissingDenseSaitsProofSignal", ...
        "Missing online proof signal: %s",name);
end
available = interp1(double(schedule.Time(:)),double(schedule.Data(:)),time,"previous")>=0.5;
used = sample(signals.saits_dense_reconstruction_used,time,1)>=0.5;
ready = sample(signals.saits_dense_model_ready,time,1)>=0.5;
validPrediction = sample(signals.saits_dense_prediction_valid,time,1)>=0.5;
measuredAvailable = sample(signals.saits_dense_measured_available,time,1)>=0.5;
backendNative = sample(signals.saits_dense_backend_native,time,1)>=0.5;
guidance = sample(signals.saits_dense_guidance_enabled,time,1)>=0.5;
reason = sample(signals.saits_dense_fallback_reason,time,1);
inferenceSeconds = sample(signals.saits_dense_inference_seconds,time,1);
rawPrediction = sample(signals.saits_dense_predicted_sectors,time,14);
controllerSectors = sample(signals.saits_dense_controller_sectors,time,14);
input = sample(signals.saits_dense_input_features,time,10752);
observedMask = sample(signals.saits_dense_input_observed_mask,time,10752);
selectedReference = sample(signals.saits_dense_selected_reference_ned_m,time,3);
waypointReference = sample(signals.saits_dense_waypoint_reference_ned_m,time,3);
referenceDelta = sample(signals.saits_dense_reference_delta_ned_m,time,3);
position = sample(signals.position_ned_m,time,3);
velocity = sample(signals.velocity_ned_mps,time,3);
oracleTargets = oracleSectorTargets(signals.nominal_observation.values, ...
    logical(signals.nominal_lidar_valid_return_mask.values));
dropout = [false;available(1:end-1) & ~available(2:end)];
dropout(1)=~available(1); % An outage already active at t=0 is one outage.
recovery = [false;~available(1:end-1) & available(2:end)];
age = zeros(n,1); lastHealthy = time(1);
for index = 1:n
    if available(index), lastHealthy = time(index); end
    age(index) = time(index)-lastHealthy;
end
normDelta = vecnorm(referenceDelta,2,2);
influenced = used & ~available & normDelta>1e-5;
proof.scanCount = n;
proof.lidarMode = lidarMode;
proof.firstScanReconstructed = used(1) && ~available(1);
proof.allScansReconstructed = all(used & ~available);
proof.noMeasuredLidarObserved = all(~available) && all(observedMask(:,1:10740)==0,"all");
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
scheduleChecks=[available(1) && available(end),nnz(dropout)==3 && nnz(recovery)==3];
scheduleLabels=["LiDAR must start and end healthy","All three outages and recoveries must be observed"];
reconstructionCoverage=any(used & ~available);
reconstructionLabel="At least one outage scan must use reconstruction";
if lidarMode=="disabled"
    scheduleChecks=[all(~available) && all(sample(signals.lidar_enabled,time,1)==0), ...
        abs(time(1))<1e-9 && nnz(dropout)==1 && nnz(recovery)==0];
    scheduleLabels=["LiDAR must remain disabled at every scan", ...
        "One continuous outage must start at t=0, with no recovery"];
    reconstructionCoverage=all(used & ~available & ready & validPrediction);
    reconstructionLabel="Every scan, including cold start at t=0, must use valid reconstruction";
end
checks = [scheduleChecks, ...
    isequal(measuredAvailable,available),all(~used | (~available & ready & validPrediction)), ...
    reconstructionCoverage,all(guidance==(available | used)), ...
    max(abs(referenceDelta-(selectedReference-waypointReference)),[],"all")<=1e-4, ...
    all(observedMask==0 | observedMask==1,"all"), ...
    all(observedMask(:,10741:10752)==1,"all") && all(observedMask(~available,1:10740)==0,"all") && all(input(~available,1:10740)==0,"all"), ...
    all(inferenceSeconds>=0)];
labels = [scheduleLabels, ...
    "Measured availability must equal the original schedule", ...
    "Reconstruction use requires outage, loaded model and valid prediction", ...
    reconstructionLabel, ...
    "Guidance enable must equal measured availability OR reconstruction use", ...
    "Reference delta must equal selected minus waypoint reference", ...
    "DENSE SAITS observed mask must be binary","Motion remains measured; unavailable LiDAR values and masks must be zero", ...
    "Inference timings must be nonnegative"];
proof.failedChecks = labels(~checks);
proof.instrumentationChecksPassed = all(checks);
scan = table(repmat(runId,n,1),uint32((1:n)'),time, ...
    'VariableNames',{'RunId','SampleIndex','TimeS'});
scan.LidarMeasuredAvailable = available;
scan.LidarDropoutEvent = dropout; scan.LidarRecoveryEvent = recovery;
scan.OutageAgeS = age; scan.DenseSaitsReconstructionUsed = used;
scan.DenseSaitsModelReady = ready; scan.DenseSaitsPredictionValid = validPrediction;
scan.DenseSaitsFallbackReason = reason; scan.DenseSaitsGuidanceEnabled = guidance;
scan.DenseSaitsInferenceSeconds = inferenceSeconds;
scan.DenseSaitsBackendNative = backendNative;
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
    scan.(sprintf("DenseSaitsPredictedSector_%02d",index)) = rawPrediction(:,index);
    scan.(sprintf("ControllerSector_%02d",index)) = controllerSectors(:,index);
    scan.(sprintf("OracleScoringTarget_%02d",index)) = oracleTargets(:,index);
end
for index = 1:12
    scan.(sprintf("DenseSaitsInput_%02d",index)) = input(:,10740+index);
    scan.(sprintf("DenseSaitsInputObserved_%02d",index)) = logical(observedMask(:,10740+index));
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
assert(size(native,2)==width,"ARES:UnexpectedDenseSaitsSignalWidth","Unexpected proof-signal width.");
tolerance = 1e-9;
assert(query(1)>=time(1)-tolerance && query(end)<=time(end)+tolerance, ...
    "ARES:DenseSaitsSignalTimeCoverage","A proof signal does not cover the scan timeline.");
query = min(max(query,time(1)),time(end));
values = interp1(time,native,query,"previous");
assert(all(isfinite(values),"all"),"ARES:NonfiniteDenseSaitsProofSignal","Proof signals must be finite.");
end

