function [manifest,telemetry,manifestPath,telemetryPath,summary] = captureSaitsBarrier(projectRoot,deploymentPath,pythonExecutable,showSimulation,continueAfterFailure)
%CAPTURESAITSBARRIER Run the prepared augmented controller and persist UUID data.
if nargin<5, continueAfterFailure=false; end
prepared=ares.inference.prepareSaitsBarrierModel(projectRoot,deploymentPath,pythonExecutable,showSimulation,continueAfterFailure);
model=prepared.modelName;
cfg=ares.config.defaultSingleUav(); cfg.sensors.lidar.enabled=true; cfg.sensors.lidar.mode="intermittent-saits-barrier";
[schedule,intervals,fault]=ares.inference.createIntermittentSchedule(double(cfg.simulation.stopTime));
manifest=ares.data.createRunManifest(projectRoot,model,"saits-barrier-intermittent-lidar-single-uav",cfg);
manifest.static.maximum_sensor_range_m=7;
manifest.static.nominal_observation_dimension=10740;
manifest.static.lidar_grid_size=[15 716];
manifest.static.observation_channel_order="elevation-index-fastest-then-azimuth-index";
manifest.static.obstacle_centers_enu_m=evalin("base","ObstaclePositions");
manifest.static.obstacle_centers_ned_m=manifest.static.obstacle_centers_enu_m(:,[2 1]);
manifest.static.obstacle_width_m=double(evalin("base","ObstacleWidth"));
manifest.static.obstacle_height_m=double(evalin("base","ObstacleHeight"));
manifest.static.oracle_usage="Raw LiDAR/geometry are scoring-only during outage; never runtime reconstruction or QP inputs.";
manifest.inference=prepared;
manifest.inference.safetyCertified=false;
manifest.groundTruth.condition="intermittent-lidar-with-augmented-saits-barrier";
manifest.groundTruth.faultType="intermittent-lidar-unavailable";
manifest.groundTruth.faultActive=true;
manifest.groundTruth.outageIntervals_s=intervals;
manifest.groundTruth.faultConfiguration=fault;
manifest.groundTruth.scheduleTime_s=double(schedule.Time(:));
manifest.groundTruth.scheduleEnabled=logical(schedule.Data(:));
fprintf("\nAugmented SAITS/BarrierNet capture %s\n",manifest.runId);
disp(array2table(intervals,"VariableNames",["StartTime_s","EndTime_s"]));
timer=tic;
simOut=sim(model);
manifest.runtime.wallSeconds=toc(timer);
manifest.runtime.completedAtUtc=string(datetime("now","TimeZone","UTC"));
% Retain even a t=0 initial rejection. Do not lose its logs because the
% baseline extractor requires multiple nominal sensor samples.
try
    telemetry=ares.data.extractRawTelemetry(simOut,manifest);
    extractionError="";
catch exception
    if ~any(string(simOut.logsout.getElementNames())=="barrier_failure_stop"), rethrow(exception); end
    stopped=simOut.logsout.getElement("barrier_failure_stop").Values;
    if ~any(stopped.Data(:)), rethrow(exception); end
    telemetry=struct("schemaVersion",manifest.schemaVersion,"runId",manifest.runId,"signals",struct());
    extractionError=string(exception.message);
end
names=string(simOut.logsout.getElementNames());
for name=names(:)'
    if name=="nominal_lidar_points_xyz_m", continue; end % NaN no-return XYZ is valid sensor data.
    element=simOut.logsout.getElement(name); series=element.Values;
    if ~isa(series,"timeseries") || isempty(series.Time) || ...
            ~(isnumeric(series.Data)||islogical(series.Data)), continue; end
    key=matlab.lang.makeValidName(name);
    if isfield(telemetry.signals,key), continue; end
    width=numel(series.Data)/numel(series.Time);
    signal=ares.data.timeseriesToSignal(series,width);
    signal.name=name; signal.unit="see runtime packet layout";
    signal.components=compose("channel_%03d",1:width);
    signal.componentUnits=repmat("mixed",1,width);
    telemetry.signals.(key)=signal;
end
oracle=simOut.logsout.getElement("nominal_lidar_points_xyz_m").Values;
telemetry.oracleLidar=struct("time_s",double(oracle.Time(:)),"pointsXyz_m",oracle.Data, ...
    "frame","sensor-FLU","usage","scoring-only during outages; never an online policy input");
packet=telemetry.signals.barrier_runtime_packet;
values=double(packet.values); times=double(packet.time_s(:));
assert(size(values,2)==113 && size(values,1)==numel(times),"ARES:BarrierLogShape","Invalid runtime packet dimensions.");
failed=values(:,110)>=.5;
available=values(:,104)>=.5;
used=values(:,105)>=.5;
feasible=values(:,106)>=.5;
assert(all(~used|~available) && all(values(~available,64:77)==0,"all"), ...
    "ARES:BarrierOracleLeak","Runtime used unavailable measured LiDAR.");
expected=interp1(double(schedule.Time),double(schedule.Data),times,"previous")>=.5;
assert(isequal(available,expected),"ARES:BarrierFaultScheduleMismatch","Runtime mask differs from seed-513 schedule.");
telemetry.mission.finalGoalPositionNed_m=cfg.mission.waypointsNed(end,:);
telemetry.mission.waypointsNed_m=cfg.mission.waypointsNed;
if strlength(extractionError)==0
    metrics=ares.data.evaluateAxisRun(telemetry,cfg,manifest.static.obstacle_centers_ned_m, ...
        manifest.static.obstacle_width_m,manifest.static.obstacle_height_m);
else
    metrics=struct("telemetryValid",false,"missionSuccess",false,"safetyChecksPassed",false, ...
        "collisionOccurred",false,"collisionEvaluated",false,"terminationReason","initial-filter-rejection", ...
        "extractionFailure",extractionError,"goalReachedAtAnyTime",false,"goalReachedAtTermination",false);
end
if any(failed)
    metrics.missionSuccess=false;
    metrics.safetyChecksPassed=false;
    if ~continueAfterFailure
        metrics.terminationReason="filter-infeasible-or-invalid";
    end
end
metrics.barrierCertificateHeldEntireRun=~any(failed);
metrics.continuedAfterBarrierFailure=logical(continueAfterFailure)&&any(failed);
metrics.uncertifiedNominalFallbackScanCount=nnz(~feasible)*logical(continueAfterFailure);
telemetry.metrics=metrics;
telemetry.inferenceMetadata=prepared;
telemetry.lidarFaultMetadata=manifest.groundTruth;
telemetry.inferenceMetadata.runtimePacketLayout=struct( ...
    "command",[1 4],"prediction",[5 18],"perceived",[19 32],"features",[33 63], ...
    "mask",[64 94],"reference",[95 97],"acceleration",[98 100],"nominal",[101 103],"diagnostics",[104 113]);
manifest.metrics=metrics;
manifest.outcome=struct("missionSuccess",metrics.missionSuccess,"safetyChecksPassed",metrics.safetyChecksPassed, ...
    "collisionOccurred",metrics.collisionOccurred,"terminationReason",metrics.terminationReason, ...
    "filterFailed",any(failed),"goalReachedAtAnyTime",metrics.goalReachedAtAnyTime, ...
    "goalReachedAtTermination",metrics.goalReachedAtTermination, ...
    "reconstructedScanCount",nnz(used),"scansLogged",numel(times));
telemetry.outcome=manifest.outcome;
manifest.inference.runtimeProof=struct("availabilityMatchesSchedule",true,"missingLidarExcluded",true, ...
    "executedController","augmented-SAITS + live BarrierNet QP + acceleration allocation", ...
    "filterFailureStopsSimulation",~logical(continueAfterFailure),"filterFailureObserved",any(failed));
subdirectory="saits_barrier_intermittent";
manifestPath=ares.data.saveRuntimeManifest(projectRoot,manifest,subdirectory);
telemetryPath=ares.data.saveRawTelemetry(projectRoot,telemetry,subdirectory);
folder=string(fileparts(telemetryPath));
columns=["command_roll_rad","command_pitch_rad","command_yaw_rate_radps","command_thrust_n", ...
    compose("predicted_lidar_%02d",1:14),compose("perceived_lidar_%02d",1:14), ...
    compose("input_feature_%02d",1:31),compose("input_observed_%02d",1:31), ...
    "reference_north_m","reference_east_m","reference_down_m", ...
    "filtered_acceleration_north","filtered_acceleration_east","filtered_acceleration_down", ...
    "nominal_acceleration_north","nominal_acceleration_east","nominal_acceleration_down", ...
    "measured_available","reconstruction_used","filter_feasible","max_constraint_violation", ...
    "tracking_slack","k2","failure_latched","inference_seconds","minimum_h","minimum_psi1"];
scans = array2table([times values],"VariableNames",["time_s" columns]);
scans = addvars(scans, repmat(manifest.runId,height(scans),1), ...
    'Before', 1, 'NewVariableNames', 'run_id');
scans.nominal_fallback_used=~feasible & logical(continueAfterFailure);

writetable(scans,fullfile(folder,"scans.csv"));
summary=struct("schemaVersion","ares-saits-barrier-run-summary-v1","runId",manifest.runId, ...
    "checkpointSha256",prepared.selectedCheckpointSha256,"metrics",metrics,"scans",numel(times), ...
    "observedOutageCount",nnz(diff(available)==-1),"reconstructedScanCount",nnz(used), ...
    "filterFeasibleScanCount",nnz(feasible),"filterFailureObserved",any(failed), ...
    "plannedOutageIntervalsS",intervals,"finalTimeS",times(end),"safetyCertified",false, ...
    "continueAfterBarrierFailure",logical(continueAfterFailure), ...
    "goalReachedAtAnyTime",metrics.goalReachedAtAnyTime,"goalReachedAtTermination",metrics.goalReachedAtTermination, ...
    "uncertifiedNominalFallbackScanCount",metrics.uncertifiedNominalFallbackScanCount);
fid=fopen(fullfile(folder,"summary.json"),"w","n","UTF-8");
assert(fid~=-1,"ARES:SummaryWriteFailed","Cannot write summary.");
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(summary,PrettyPrint=true));
assert(ares.inference.fileSha256(fullfile(projectRoot,"models","ares_single_uav_top.slx"))==prepared.baselineModelSha256, ...
    "ARES:BaselineChanged","Baseline changed during capture.");
fprintf("Telemetry saved to %s\n",folder);
fprintf("Filter failure: %d. Mission success: %d. Reconstructed scans: %d.\n",any(failed),metrics.missionSuccess,nnz(used));
fprintf("Goal reached: %d. Nominal fallback scans: %d.\n",metrics.goalReachedAtAnyTime,metrics.uncertifiedNominalFallbackScanCount);
end
