function [batchPath, index] = captureLidarAxisBatch(projectRoot, options)
%CAPTURELIDARAXISBATCH Collect 40 North-axis and 40 East-axis healthy runs.
% Raw runs use existing UUID/manifests and immutable writers. Mission failures
% are retained; simulator/instrumentation failures have explicit attempt logs.
% Resume by passing ResumeBatchPath. Completed artifacts are never overwritten.
arguments
    projectRoot (1,1) string
    options.Seed (1,1) double {mustBeInteger,mustBeNonnegative} = 20261102
    options.OutputRoot (1,1) string = ""
    options.ResumeBatchPath (1,1) string = ""
    options.RunIndices (1,:) double {mustBeInteger,mustBePositive} = 1:80
    options.WorkerId (1,1) string = ""
    options.PrepareOnly (1,1) logical = false
end
assert(isfile(fullfile(projectRoot,"ares.prj")),"ARES:InvalidProject", ...
    "ARES project not found: %s", projectRoot);
assert(all(options.RunIndices<=80),"ARES:InvalidRunIndex","Run indices must be 1:80.");
assert(options.WorkerId=="" || ~isempty(regexp(options.WorkerId,"^[a-zA-Z0-9_]+$","once")), ...
    "ARES:InvalidWorkerId","Worker ID must contain only letters, numbers and underscores.");
assert(options.WorkerId=="" || options.ResumeBatchPath~="", ...
    "ARES:WorkerRequiresBatch","Prepare the batch before starting independent workers.");
addpath(fullfile(projectRoot,"src"),fullfile(projectRoot,"scripts"),"-end");
outputRoot=options.OutputRoot;
if outputRoot=="", outputRoot=projectRoot; end
model="ares_single_uav_top";
assert(~bdIsLoaded(model),"ARES:ModelAlreadyLoaded", ...
    "Close the model before collecting; the batch loads the saved model and closes its temporary logging changes.");
cfg=ares.config.defaultSingleUav();
assert(isequal(cfg.mission.waypointsNed(end,:),[30 35 -7]) && ...
    isequal(cfg.mission.initialPositionNed,[0 0 -7]), ...
    "ARES:ScenarioChanged","Review sampling bounds after changing the default mission.");
sourceSnapshot=snapshotSources(projectRoot);
if options.ResumeBatchPath==""
    plan=ares.data.createAxisStartPlan(options.Seed);
    batchId=string(java.util.UUID.randomUUID());
    batchPath=fullfile(outputRoot,"data","batches",batchId);
    assert(~isfolder(batchPath),"ARES:BatchAlreadyExists","Batch folder already exists.");
    mkdir(batchPath); mkdir(fullfile(batchPath,"attempts"));
    metadata.schemaVersion="lidar-axis-batch-v1";
    metadata.batchId=batchId;
    metadata.createdAtUtc=utcNow();
    metadata.randomSeed=options.Seed;
    metadata.outputRoot=string(java.io.File(outputRoot).getCanonicalPath());
    metadata.plannedRunCount=80;
    metadata.axisInterpretation="North [N,0,-7], East [0,E,-7]; positive mission extents";
    metadata.sampling="40 stratified random points per feasible axis; no repeated starts";
    metadata.sourceSnapshot=sourceSnapshot;
    metadata.configuration=cfg;
    metadata.coordinateNote="Initializer polygon coordinates are ENU; NED centers swap columns; scene unchanged.";
    metadata.obstacleCentersEnuM=[10 0;15 15;10 20;5 15;22 18;15 25;10 20];
    metadata.obstacleCentersNedM=metadata.obstacleCentersEnuM(:,[2 1]);
    metadata.obstacleWidthM=3; metadata.obstacleHeightM=15;
    metadata.noiseNote="Model LiDAR block HasNoise=off and RangeAccuracy=0.02; effective parameters recorded per run.";
    metadata.chatReference="User authorized proceeding without inaccessible shared chat.";
    metadata.splitPolicy="No fitted preprocessing or row splits. Split complete run IDs before windowing/calibration.";
    save(fullfile(batchPath,"plan.mat"),"plan","metadata");
    writetable(plan,fullfile(batchPath,"plan.csv"));
    writeJson(fullfile(batchPath,"batch_manifest.json"),metadata);
    index=plan;
    count=height(index);
    index.RunId=strings(count,1); index.Status=repmat("pending",count,1);
    index.AttemptCount=zeros(count,1);
    index.RawTelemetryPath=strings(count,1); index.EpisodePath=strings(count,1);
    index.ScanTablePath=strings(count,1);
    index.MissionSuccess=false(count,1); index.CollisionOccurred=false(count,1);
    index.SafetyChecksPassed=false(count,1); index.ScanCount=zeros(count,1);
    index.DurationS=nan(count,1); index.FinalWaypointErrorM=nan(count,1);
    index.MinimumSurfaceClearanceM=nan(count,1); index.ValidReturnFraction=nan(count,1);
    index.TerminationReason=strings(count,1); index.FailureIdentifier=strings(count,1);
    index.FailureMessage=strings(count,1);
    checkpoint(batchPath,index,options.WorkerId);
else
    batchPath=options.ResumeBatchPath;
    saved=load(fullfile(batchPath,"plan.mat"),"plan","metadata");
    plan=saved.plan; metadata=saved.metadata;
    amendments=dir(fullfile(batchPath,"source_amendment_*.mat"));
    for amendmentIndex=1:numel(amendments)
        record=load(fullfile(amendments(amendmentIndex).folder,amendments(amendmentIndex).name),"amendment");
        amendment=record.amendment;
        assert(string(amendment.batchId)==string(metadata.batchId) && ...
            isequal(amendment.fromSnapshot,metadata.sourceSnapshot), ...
            "ARES:InvalidSourceAmendment","Source amendment chain is invalid.");
        metadata.sourceSnapshot=amendment.toSnapshot;
    end
    if options.OutputRoot~=""
        assert(string(java.io.File(options.OutputRoot).getCanonicalPath())==metadata.outputRoot, ...
            "ARES:ResumeOutputRootMismatch","Resume output root must match the original batch.");
    end
    outputRoot=metadata.outputRoot;
    checkpointPath=fullfile(batchPath,"checkpoint"+workerSuffix(options.WorkerId)+".mat");
    if ~isfile(checkpointPath), checkpointPath=fullfile(batchPath,"checkpoint.mat"); end
    state=load(checkpointPath,"index"); index=state.index;
    assert(isequal(sourceSnapshot,metadata.sourceSnapshot),"ARES:BatchSourceChanged", ...
        "Source files changed since this batch was planned; begin a new batch.");
    assert(height(index)==80 && isequal(index.RunIndex,plan.RunIndex), ...
        "ARES:InvalidCheckpoint","Checkpoint does not match the 80-run plan.");
end
fprintf("BATCH_PATH %s\n",batchPath);
if options.PrepareOnly, return; end
load_system(fullfile(projectRoot,"models",model+".slx"));
cleanup=onCleanup(@()close_system(model,0));
% This sink only draws figures. Disable it in memory; never save the model.
set_param(model+"/UAV Scenario Scope","Commented","on");
instrumentState(model);
for row=options.RunIndices
    if index.Status(row)=="complete"
        verifyRun(index(row,:));
        fprintf("SKIP_COMPLETE %d %s\n",row,index.RunId(row));
        continue;
    end
    index.AttemptCount(row)=index.AttemptCount(row)+1;
    attemptNumber=index.AttemptCount(row);
    runCfg=cfg;
    runCfg.sensors.lidar.enabled=true;
    start=[plan.PositionNorthM(row),plan.PositionEastM(row),plan.PositionDownM(row)];
    runCfg.mission.initialPositionNed=start;
    runCfg.mission.waypointsNed(1,:)=start;
    manifest=ares.data.createRunManifest(projectRoot,model,"lidar-on-axis-single-uav",runCfg);
    manifest.experiment.batchId=metadata.batchId;
    manifest.experiment.runIndex=row;
    manifest.experiment.axis=plan.Axis(row);
    manifest.experiment.axisIndex=plan.AxisIndex(row);
    manifest.experiment.randomSeed=plan.SimulationSeed(row);
    manifest.experiment.samplingSeed=metadata.randomSeed;
    manifest.experiment.attempt=attemptNumber;
    manifest.experiment.sourceAmendmentCount=numel(dir(fullfile(batchPath,"source_amendment_*.mat")));
    manifest.experiment.scenarioType="healthy-lidar-axis-start";
    manifest.groundTruth.condition="nominal-lidar-on";
    manifest.groundTruth.faultType="none";
    manifest.groundTruth.faultActive=false;
    manifest.groundTruth.expectedMissionOutcome="reach-goal-without-collision";
    manifest.static.maximum_sensor_range_m=7;
    manifest.static.nominal_observation_schema="organized-lidar-range-vector-v1";
    manifest.static.nominal_observation_dimension=10740;
    manifest.static.lidar_grid_size=[15 716];
    manifest.static.observation_channel_order="elevation-index-fastest-then-azimuth-index";
    manifest.static.invalid_lidar_return_encoding="maximum-range-with-validity-mask";
    manifest.static.initial_position_ned_m=start;
    manifest.static.waypoints_ned_m=runCfg.mission.waypointsNed;
    manifest.static.obstacle_centers_enu_m=metadata.obstacleCentersEnuM;
    manifest.static.obstacle_centers_ned_m=metadata.obstacleCentersNedM;
    manifest.static.obstacle_width_m=metadata.obstacleWidthM;
    manifest.static.obstacle_height_m=metadata.obstacleHeightM;
    manifest.static.obstacle_mesh_reference_frame="ENU";
    manifest.static.lidar_mounting_location_m=[0 0 -0.4];
    manifest.static.lidar_mounting_angles_deg=[0 0 180];
    manifest.static.lidar_points_frame="sensor frame: x forward, y left, z up";
    manifest.static.state_truth_note="Simulator ground truth; no estimator is present.";
    manifest.provenance=metadata.sourceSnapshot;
    index.RunId(row)=manifest.runId; index.Status(row)="running";
    index.RawTelemetryPath(row)=""; index.EpisodePath(row)=""; index.ScanTablePath(row)="";
    index.MissionSuccess(row)=false; index.CollisionOccurred(row)=false;
    index.SafetyChecksPassed(row)=false; index.ScanCount(row)=0;
    index.DurationS(row)=NaN; index.FinalWaypointErrorM(row)=NaN;
    index.MinimumSurfaceClearanceM(row)=NaN; index.ValidReturnFraction(row)=NaN;
    index.TerminationReason(row)="";
    index.FailureIdentifier(row)=""; index.FailureMessage(row)="";
    checkpoint(batchPath,index,options.WorkerId);
    fprintf("RUN_START %d/80 axis=%s start=%s id=%s\n",row,plan.Axis(row),mat2str(start,6),manifest.runId);
    wallStart=tic;
    stage="initialize"; failureReport="";
    try
        rng(plan.SimulationSeed(row),"twister");
        assignin("base","SingleUavConfigOverride",runCfg);
        evalin("base","clear SingleUavLidarScheduleOverride;");
        initializer=replace(fullfile(projectRoot,"scripts","initializeSingleUav.m"),"'","''");
        evalin("base","run('"+initializer+"');");
        assert(isequal(evalin("base","ObstaclePositions"),metadata.obstacleCentersEnuM) && ...
            evalin("base","ObstacleWidth")==metadata.obstacleWidthM && ...
            evalin("base","ObstacleHeight")==metadata.obstacleHeightM, ...
            "ARES:ScenarioGeometryChanged","Initializer geometry differs from the sampling/safety snapshot.");
        schedule=evalin("base","LidarEnableSchedule");
        assert(all(schedule.Data==1),"ARES:UnexpectedLidarOutage","Expected always-on LiDAR schedule.");
        set_param(model,"StopTime",num2str(runCfg.simulation.stopTime));
        manifest.static.lidar_effective_block_parameters=lidarParameters(model);
        stage="simulate";
        simOut=sim(model);
        stage="extract";
        telemetry=ares.data.extractRawTelemetry(simOut,manifest);
        telemetry=extractExtraState(telemetry,simOut);
        telemetry.mission.finalGoalPositionNed_m=runCfg.mission.waypointsNed(end,:);
        telemetry.mission.waypointsNed_m=runCfg.mission.waypointsNed;
        telemetry.mission.desiredPositionSignalMeaning="avoidance-adjusted-controller-reference";
        rawCloud=simOut.logsout.getElement("nominal_lidar_points_xyz_m").Values;
        telemetry.rawLidar.time_s=double(rawCloud.Time(:));
        telemetry.rawLidar.pointsXyz_m=rawCloud.Data;
        telemetry.rawLidar.dimensionOrder="elevation,azimuth,xyz,scan";
        telemetry.rawLidar.frame="sensor frame: x forward, y left, z up";
        metrics=ares.data.evaluateAxisRun(telemetry,runCfg,metadata.obstacleCentersNedM,3,15);
        assert(metrics.telemetryValid && metrics.initialPositionMatchesPlan, ...
            "ARES:InvalidRunTelemetry","Telemetry is invalid or actual spawn differs from planned spawn.");
        enabled=telemetry.signals.lidar_enabled.values;
        assert(all(enabled==1,"all"),"ARES:UnexpectedLidarOutage","LiDAR was disabled during the run.");
        scanTime=telemetry.signals.nominal_observation.time_s;
        assert(numel(scanTime)>=2 && all(diff(scanTime)>0) && ...
            max(abs(diff(scanTime)-0.1))<1e-6, ...
            "ARES:InvalidScanCadence","Expected a continuous 10 Hz scan timeline.");
        manifest.metrics=metrics;
        manifest.outcome=struct("safetyChecksPassed",metrics.safetyChecksPassed, ...
            "missionSuccess",metrics.missionSuccess,"collisionOccurred",metrics.collisionOccurred, ...
            "terminationReason",metrics.terminationReason,"instrumentationValid",true, ...
            "failureIdentifier","","failureMessage","");
        manifest.runtime.wallSeconds=toc(wallStart);
        manifest.runtime.simulationSampleCount=numel(simOut.tout);
        manifest.runtime.completedAtUtc=utcNow();
        telemetry.experiment=manifest.experiment;
        telemetry.outcome=manifest.outcome;
        telemetry.metrics=metrics;
        stage="save-raw";
        ares.data.saveRuntimeManifest(outputRoot,manifest);
        telemetryPath=ares.data.saveRawTelemetry(outputRoot,telemetry);
        index.RawTelemetryPath(row)=telemetryPath;
        processedFolder=fullfile(outputRoot,"data","processed","lidar_episode_v1",manifest.runId);
        episodePath=fullfile(processedFolder,"lidar-on-axis-single-uav_lidar_episode.mat");
        index.EpisodePath(row)=episodePath;
        stage="build-episode";
        episode=buildLidarEpisode(telemetryPath,episodePath);
        scanPath=fullfile(processedFolder,"scans.csv");
        index.ScanTablePath(row)=scanPath;
        writeScanTable(episode,manifest,scanPath);
        index.EpisodePath(row)=episodePath; index.ScanTablePath(row)=scanPath;
        index.MissionSuccess(row)=metrics.missionSuccess;
        index.CollisionOccurred(row)=metrics.collisionOccurred;
        index.SafetyChecksPassed(row)=metrics.safetyChecksPassed;
        index.ScanCount(row)=numel(episode.time_s);
        index.DurationS(row)=metrics.durationS;
        index.FinalWaypointErrorM(row)=metrics.finalWaypointErrorM;
        index.MinimumSurfaceClearanceM(row)=metrics.minimumObstacleClearanceM;
        index.ValidReturnFraction(row)=episode.metadata.validReturnFraction;
        index.TerminationReason(row)=metrics.terminationReason;
        verifyRun(index(row,:));
        index.Status(row)="complete";
        stage="complete";
        fprintf("RUN_COMPLETE %d/80 scans=%d success=%d collision=%d wall=%.1fs\n", ...
            row,index.ScanCount(row),metrics.missionSuccess,metrics.collisionOccurred,toc(wallStart));
    catch failure
        index.Status(row)="instrumentation-failed";
        index.FailureIdentifier(row)=string(failure.identifier);
        index.FailureMessage(row)=string(failure.message);
        failureReport=string(getReport(failure,"extended","hyperlinks","off"));
        fprintf(2,"RUN_FAILED %d stage=%s %s: %s\n",row,stage,failure.identifier,failure.message);
    end
    attempt=struct("runIndex",row,"runId",index.RunId(row),"attempt",attemptNumber, ...
        "status",index.Status(row),"stage",stage,"completedAtUtc",utcNow(), ...
        "wallSeconds",toc(wallStart),"failureIdentifier",index.FailureIdentifier(row), ...
        "failureMessage",index.FailureMessage(row),"failureReport",failureReport, ...
        "rawTelemetryPath",index.RawTelemetryPath(row),"episodePath",index.EpisodePath(row), ...
        "scanTablePath",index.ScanTablePath(row));
    writeJson(fullfile(batchPath,"attempts",sprintf("run_%03d_attempt_%02d.json",row,attemptNumber)),attempt);
    checkpoint(batchPath,index,options.WorkerId);
    clear simOut telemetry episode;
    % Stop on instrumentation trouble; do not spend 80 runs repeating a defect.
    if index.Status(row)~="complete", break; end
end
summary=struct("batchId",metadata.batchId,"plannedRuns",80, ...
    "completedRuns",nnz(index.Status=="complete"), ...
    "instrumentationFailures",nnz(index.Status=="instrumentation-failed"), ...
    "missionSuccesses",nnz(index.MissionSuccess & index.Status=="complete"), ...
    "collisions",nnz(index.CollisionOccurred & index.Status=="complete"), ...
    "totalScans",sum(index.ScanCount),"updatedAtUtc",utcNow());
replaceJson(fullfile(batchPath,"summary"+workerSuffix(options.WorkerId)+".json"),summary);
fprintf("BATCH_PROGRESS completed=%d/80 scans=%d\n",summary.completedRuns,summary.totalScans);
end

function instrumentState(model)
selectors=find_system(model,"LookUnderMasks","all","BlockType","BusSelector");
selector="";
for k=1:numel(selectors)
    if string(get_param(selectors{k},"OutputSignals"))=="WorldPosition,EulerZYX"
        selector=string(selectors{k}); break;
    end
end
assert(selector~="","ARES:MissingStateBus","Cannot locate plant state bus.");
ports=get_param(selector,"PortHandles");
set_param(ports.Outport(2),"DataLogging","on","DataLoggingNameMode","Custom", ...
    "DataLoggingName","batch_attitude_zyx_rad");
parent=string(get_param(selector,"Parent"));
source=get_param(get_param(ports.Inport(1),"Line"),"SrcPortHandle");
angular=parent+"/ARES Batch Angular Rate";
sink=parent+"/ARES Batch Angular Logger";
add_block("simulink/Signal Routing/Bus Selector",angular, ...
    "OutputSignals","BodyAngularRateRPY","Position",[440 310 460 350]);
add_block("simulink/Sinks/To Workspace",sink,"VariableName","batch_angular_velocity_body_radps", ...
    "SaveFormat","Timeseries","MaxDataPoints","inf","Position",[520 310 680 350]);
angularPorts=get_param(angular,"PortHandles"); sinkPorts=get_param(sink,"PortHandles");
add_line(parent,source,angularPorts.Inport(1),"autorouting","on");
add_line(parent,angularPorts.Outport(1),sinkPorts.Inport(1),"autorouting","on");
end

function telemetry=extractExtraState(telemetry,simOut)
attitude=ares.data.timeseriesToSignal(simOut.logsout.getElement("batch_attitude_zyx_rad").Values,3);
attitude.values=attitude.values(:,[3 2 1]);
attitude.name="attitude_ned_rpy_rad"; attitude.unit="rad";
attitude.components=["roll","pitch","yaw"];
telemetry.signals.attitude_ned_rpy_rad=attitude;
angular=ares.data.timeseriesToSignal(simOut.batch_angular_velocity_body_radps,3);
angular.name="angular_velocity_body_radps"; angular.unit="rad/s";
angular.components=["roll_rate","pitch_rate","yaw_rate"];
telemetry.signals.angular_velocity_body_radps=angular;
telemetry.contractCoverage.availableLoggedSignals(end+1:end+2)=[ ...
    "attitude_ned_rpy_rad";"angular_velocity_body_radps"];
end

function parameters=lidarParameters(model)
block=model+"/UAV Scenario Lidar";
names=["SensorName","MaxRange","RangeAccuracy","AzimuthLimits", ...
    "AzimuthResolution","ElevationLimits","ElevationResolution", ...
    "HasNoise","HasOrganizedOutput","SampleTime"];
parameters=struct();
for name=names
    expression=string(get_param(block,name));
    parameters.(name).expression=expression;
    try, parameters.(name).value=evalin("base",expression); ...
    catch, parameters.(name).value=expression; end
end
end

function writeScanTable(episode,manifest,path)
assert(~isfile(path),"ARES:ScanTableAlreadyExists","Scan table already exists.");
n=numel(episode.time_s);
valid=episode.lidar.validReturnMask;
ranges=episode.lidar.ranges_m; ranges(~valid)=NaN;
nearest=min(ranges,[],2,"omitnan"); nearest(all(~valid,2))=NaN;
scan=table(repmat(episode.runId,n,1),repmat(manifest.experiment.axis,n,1), ...
    episode.sampleIndex,episode.time_s, ...
    'VariableNames',{'RunId','Axis','SampleIndex','TimeS'});
scan.PositionNorthM=episode.state.positionNed_m(:,1);
scan.PositionEastM=episode.state.positionNed_m(:,2);
scan.PositionDownM=episode.state.positionNed_m(:,3);
scan.VelocityNorthMps=episode.state.velocityNed_mps(:,1);
scan.VelocityEastMps=episode.state.velocityNed_mps(:,2);
scan.VelocityDownMps=episode.state.velocityNed_mps(:,3);
scan.RollRad=episode.state.attitudeNedRpy_rad(:,1);
scan.PitchRad=episode.state.attitudeNedRpy_rad(:,2);
scan.YawRad=episode.state.attitudeNedRpy_rad(:,3);
scan.BodyRollRateRadps=episode.state.angularVelocityBody_radps(:,1);
scan.BodyPitchRateRadps=episode.state.angularVelocityBody_radps(:,2);
scan.BodyYawRateRadps=episode.state.angularVelocityBody_radps(:,3);
scan.GoalNorthM=episode.goal.positionNed_m(:,1);
scan.GoalEastM=episode.goal.positionNed_m(:,2);
scan.GoalDownM=episode.goal.positionNed_m(:,3);
scan.DesiredReferenceNorthM=episode.reference.desiredPositionNed_m(:,1);
scan.DesiredReferenceEastM=episode.reference.desiredPositionNed_m(:,2);
scan.DesiredReferenceDownM=episode.reference.desiredPositionNed_m(:,3);
scan.CommandRollRad=episode.action.current(:,1);
scan.CommandPitchRad=episode.action.current(:,2);
scan.CommandYawRateRadps=episode.action.current(:,3);
scan.CommandThrustN=episode.action.current(:,4);
scan.PreviousActionAvailable=episode.action.previousAvailable;
scan.LidarEnabled=episode.lidar.enabled;
scan.ValidReturnFraction=sum(valid,2)/size(valid,2);
scan.NearestValidReturnM=nearest;
scan.DistanceToFinalGoalM=vecnorm(episode.state.positionNed_m-manifest.configuration.mission.waypointsNed(end,:),2,2);
scan.ObstacleAvoidanceStatus=episode.obstacleAvoidanceStatus;
writetable(scan,path);
end

function verifyRun(row)
assert(isfile(row.RawTelemetryPath) && isfile(row.EpisodePath) && isfile(row.ScanTablePath), ...
    "ARES:MissingRunArtifact","Completed run has a missing artifact.");
manifest=jsondecode(fileread(fullfile(fileparts(row.RawTelemetryPath),"manifest.json")));
raw=load(row.RawTelemetryPath,"telemetry"); processed=load(row.EpisodePath,"episode");
assert(string(manifest.runId)==row.RunId && string(raw.telemetry.runId)==row.RunId && ...
    string(processed.episode.runId)==row.RunId, ...
    "ARES:RunIdMismatch","Manifest, telemetry and episode IDs must match.");
assert(all(raw.telemetry.signals.lidar_enabled.values==1,"all") && ...
    numel(processed.episode.time_s)==row.ScanCount, ...
    "ARES:InvalidCompletedRun","Completed run failed read-back checks.");
end

function checkpoint(folder,index,workerId)
suffix=workerSuffix(workerId);
temp=fullfile(folder,"checkpoint"+suffix+".tmp.mat"); save(temp,"index");
movefile(temp,fullfile(folder,"checkpoint"+suffix+".mat"),"f");
temp=fullfile(folder,"index"+suffix+".tmp.csv"); writetable(index,temp);
movefile(temp,fullfile(folder,"index"+suffix+".csv"),"f");
end

function suffix=workerSuffix(workerId)
suffix="";
if workerId~="", suffix="_"+workerId; end
end

function snapshot=snapshotSources(root)
relative=["models/ares_single_uav_top.slx","scripts/initializeSingleUav.m", ...
    "scripts/setupSingleUavBusObjects.m", ...
    "src/+ares/+config/defaultSingleUav.m","src/+ares/+data/extractRawTelemetry.m", ...
    "src/+ares/+data/defaultDataContract.m","src/+ares/+data/timeseriesToSignal.m", ...
    "src/+ares/+data/createRunManifest.m","src/+ares/+data/saveRuntimeManifest.m", ...
    "src/+ares/+data/saveRawTelemetry.m"];
snapshot=struct();
snapshot.matlabVersion=string(version);
snapshot.platform=string(computer);
snapshot.toolboxes=ver;
for k=1:numel(relative)
    snapshot.files(k).path=relative(k);
    snapshot.files(k).sha256=fileHash(fullfile(root,relative(k)));
end
functions=["captureLidarAxisBatch","buildLidarEpisode","ares.data.createAxisStartPlan","ares.data.evaluateAxisRun"];
for k=1:numel(functions)
    snapshot.collectionCode(k).name=functions(k);
    snapshot.collectionCode(k).sha256=fileHash(string(which(functions(k))));
end
end

function hash=fileHash(path)
fid=fopen(path,"rb"); assert(fid~=-1,"ARES:SourceReadFailed","Cannot read source.");
cleanup=onCleanup(@()fclose(fid));
digest=java.security.MessageDigest.getInstance("SHA-256");
while ~feof(fid), bytes=fread(fid,1048576,"*uint8"); digest.update(bytes); end
hash=lower(string(reshape(dec2hex(typecast(digest.digest(),"uint8"),2).',1,[])));
end

function value=utcNow()
value=string(datetime("now","TimeZone","UTC","Format","yyyy-MM-dd'T'HH:mm:ss.SSSXXX"));
end

function writeJson(path,value)
assert(~isfile(path),"ARES:ArtifactAlreadyExists","Artifact already exists: %s",path);
fid=fopen(path,"w","n","UTF-8"); assert(fid~=-1,"ARES:WriteFailed","Cannot write: %s",path);
cleanup=onCleanup(@()fclose(fid)); fprintf(fid,"%s",jsonencode(value,PrettyPrint=true));
end

function replaceJson(path,value)
temp=path+".tmp";
fid=fopen(temp,"w","n","UTF-8"); assert(fid~=-1,"ARES:WriteFailed","Cannot write: %s",temp);
fprintf(fid,"%s",jsonencode(value,PrettyPrint=true)); fclose(fid);
movefile(temp,path,"f");
end
