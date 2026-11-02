function recovered = recoverLidarAxisExports(batchPath)
%RECOVERLIDARAXISEXPORTS Re-export saved runs affected by endpoint roundoff.
% Does not simulate, allocate a UUID, overwrite raw data, or change attempts.
% Original failures remain in immutable attempt records; new recovery records
% identify the exact raw file and builder used to complete each export.
arguments
    batchPath (1,1) string
end
saved = load(fullfile(batchPath,"plan.mat"),"plan","metadata");
plan = saved.plan; metadata = saved.metadata;
files = dir(fullfile(batchPath,"checkpoint_worker*.mat"));
assert(~isempty(files),"ARES:MissingWorkerCheckpoints","No worker checkpoints found.");
builderPath = string(which("buildLidarEpisode"));
assert(isfile(builderPath),"ARES:MissingEpisodeBuilder","Cannot locate buildLidarEpisode.");
builderSha256 = fileHash(builderPath);
recovered = zeros(0,1);
for fileIndex = 1:numel(files)
    checkpointPath = fullfile(files(fileIndex).folder,files(fileIndex).name);
    state = load(checkpointPath,"index"); index = state.index;
    assert(istable(index) && height(index)==80 && ...
        isequal(index(:,plan.Properties.VariableNames),plan), ...
        "ARES:InvalidCheckpoint","Worker checkpoint differs from frozen plan.");
    rows = find(string(index.Status)=="instrumentation-failed" & ...
        string(index.FailureIdentifier)=="ARES:ResamplingOutsideTimeRange");
    for row = rows'
        assert(~ismember(row,recovered),"ARES:DuplicateRecoveryCandidate", ...
            "Run %d appears as a failed export in multiple worker checkpoints.",row);
        original = index(row,:);
        rawPath = string(original.RawTelemetryPath);
        assert(isfile(rawPath),"ARES:MissingSavedRawRun", ...
            "Cannot recover run %d: its raw telemetry was not saved.",row);
        manifestPath = fullfile(fileparts(rawPath),"manifest.json");
        assert(isfile(manifestPath),"ARES:MissingSavedManifest","Saved manifest is missing.");
        manifest = jsondecode(fileread(manifestPath));
        loaded = load(rawPath,"telemetry"); telemetry = loaded.telemetry;
        id = string(original.RunId);
        assert(strlength(id)>0 && string(manifest.runId)==id && ...
            string(telemetry.runId)==id && ...
            string(manifest.experiment.batchId)==string(metadata.batchId) && ...
            manifest.experiment.runIndex==row && ...
            string(manifest.experiment.axis)==string(plan.Axis(row)), ...
            "ARES:RecoveryRunIdMismatch","Saved run identity differs from checkpoint/plan.");
        assert(all(telemetry.signals.lidar_enabled.values==1,"all") && ...
            telemetry.metrics.telemetryValid && telemetry.metrics.initialPositionMatchesPlan, ...
            "ARES:InvalidRecoveryTelemetry","Native saved telemetry is not a healthy LiDAR run.");
        assert(isfield(telemetry,"rawLidar") && ...
            isfield(telemetry.rawLidar,"pointsXyz_m"), ...
            "ARES:MissingSavedPointClouds","Saved native XYZ point clouds are missing.");
        rawSha256 = fileHash(rawPath);
        manifestSha256 = fileHash(manifestPath);
        episodePath = string(original.EpisodePath);
        assert(strlength(episodePath)>0,"ARES:MissingPlannedEpisodePath", ...
            "Original episode path is missing; refusing to choose a new location.");
        episodeExisted = isfile(episodePath);
        if episodeExisted
            processed = load(episodePath,"episode"); episode = processed.episode;
        else
            episode = buildLidarEpisode(rawPath,episodePath);
        end
        assert(string(episode.runId)==id && ...
            string(episode.sourceTelemetryPath)==rawPath && ...
            isequal(episode.time_s(:),telemetry.signals.nominal_observation.time_s(:)) && ...
            size(episode.lidar.ranges_m,2)==10740 && ...
            isequal(size(episode.lidar.ranges_m),size(episode.lidar.validReturnMask)) && ...
            all(episode.lidar.enabled) && ...
            all(isfinite(episode.lidar.ranges_m),"all"), ...
            "ARES:InvalidRecoveredEpisode","Processed artifact failed recovery checks.");
        count = numel(episode.time_s);
        assert(count>=2 && all(diff(episode.time_s)>0) && ...
            max(abs(diff(episode.time_s)-0.1))<1e-6, ...
            "ARES:InvalidRecoveredCadence","Recovered scans are not at 10 Hz.");
        scanPath = fullfile(fileparts(episodePath),"scans.csv");
        if strlength(string(original.ScanTablePath))>0
            assert(string(original.ScanTablePath)==scanPath, ...
                "ARES:RecoveryScanPathMismatch","Checkpoint scan CSV path differs from planned episode folder.");
        end
        expectedScan = makeScanTable(episode,manifest);
        scanExisted = isfile(scanPath);
        if scanExisted
            actualScan = readtable(scanPath,"TextType","string");
            verifyScanTable(actualScan,expectedScan);
        else
            temporary = scanPath+".recovery.tmp.csv";
            assert(~isfile(temporary),"ARES:RecoveryTemporaryFileExists", ...
                "A prior incomplete recovery CSV exists: %s",temporary);
            writetable(expectedScan,temporary);
            movefile(temporary,scanPath);
            actualScan = readtable(scanPath,"TextType","string");
            verifyScanTable(actualScan,expectedScan);
        end
        metrics = telemetry.metrics;
        index.EpisodePath(row) = episodePath;
        index.ScanTablePath(row) = scanPath;
        index.ScanCount(row) = count;
        index.MissionSuccess(row) = metrics.missionSuccess;
        index.CollisionOccurred(row) = metrics.collisionOccurred;
        index.SafetyChecksPassed(row) = metrics.safetyChecksPassed;
        index.DurationS(row) = metrics.durationS;
        index.FinalWaypointErrorM(row) = metrics.finalWaypointErrorM;
        index.MinimumSurfaceClearanceM(row) = metrics.minimumObstacleClearanceM;
        index.ValidReturnFraction(row) = episode.metadata.validReturnFraction;
        index.TerminationReason(row) = metrics.terminationReason;
        index.Status(row) = "complete";
        index.FailureIdentifier(row) = "";
        index.FailureMessage(row) = "";
        assert(index.AttemptCount(row)==original.AttemptCount && ...
            index.RunId(row)==original.RunId && ...
            index.RawTelemetryPath(row)==original.RawTelemetryPath, ...
            "ARES:RecoveryChangedIdentity","Recovery changed run identity or attempt count.");
        assert(fileHash(rawPath)==rawSha256 && fileHash(manifestPath)==manifestSha256, ...
            "ARES:RecoveryMutatedRawData","Immutable raw telemetry or manifest changed during export.");
        recovery = struct("schemaVersion","lidar-axis-export-recovery-v1", ...
            "batchId",string(metadata.batchId),"runIndex",row,"runId",id, ...
            "recoveredAtUtc",utcNow(),"workerCheckpoint",string(files(fileIndex).name), ...
            "originalStatus",string(original.Status), ...
            "originalFailureIdentifier",string(original.FailureIdentifier), ...
            "originalFailureMessage",string(original.FailureMessage), ...
            "attemptCountPreserved",original.AttemptCount, ...
            "operation","rebuild derived episode and scan summary from original saved raw telemetry", ...
            "simulationRerun",false,"newRunIdAllocated",false, ...
            "rawTelemetryPath",rawPath,"rawTelemetrySha256",rawSha256, ...
            "manifestPath",string(manifestPath),"manifestSha256",manifestSha256, ...
            "episodePath",episodePath,"scanTablePath",string(scanPath), ...
            "episodeReused",episodeExisted,"scanTableReused",scanExisted, ...
            "episodeBuilderPath",builderPath,"episodeBuilderSha256",builderSha256, ...
            "originalRunProvenance",manifest.provenance, ...
            "scanCount",count,"missionOutcomePreserved",telemetry.outcome);
        recoveryFolder = fullfile(batchPath,"recoveries");
        if ~isfolder(recoveryFolder), mkdir(recoveryFolder); end
        recoveryIndex = 1;
        recoveryPath = fullfile(recoveryFolder,sprintf("run_%03d_recovery_%02d.json",row,recoveryIndex));
        while isfile(recoveryPath)
            recoveryIndex = recoveryIndex+1;
            recoveryPath = fullfile(recoveryFolder,sprintf("run_%03d_recovery_%02d.json",row,recoveryIndex));
        end
        writeImmutableJson(recoveryPath,recovery);
        atomicWorkerCheckpoint(checkpointPath,index);
        recovered(end+1,1) = row; %#ok<AGROW>
        fprintf("EXPORT_RECOVERED run=%d id=%s scans=%d checkpoint=%s\n", ...
            row,id,count,files(fileIndex).name);
        clear loaded telemetry processed episode actualScan expectedScan;
    end
end
fprintf("EXPORT_RECOVERY_COMPLETE count=%d\n",numel(recovered));
end

function scan = makeScanTable(episode,manifest)
% Deliberately matches captureLidarAxisBatch.writeScanTable column for column.
n = numel(episode.time_s);
valid = episode.lidar.validReturnMask;
ranges = episode.lidar.ranges_m; ranges(~valid) = NaN;
nearest = min(ranges,[],2,"omitnan"); nearest(all(~valid,2)) = NaN;
scan = table(repmat(episode.runId,n,1),repmat(string(manifest.experiment.axis),n,1), ...
    episode.sampleIndex,episode.time_s, ...
    'VariableNames',{'RunId','Axis','SampleIndex','TimeS'});
scan.PositionNorthM = episode.state.positionNed_m(:,1);
scan.PositionEastM = episode.state.positionNed_m(:,2);
scan.PositionDownM = episode.state.positionNed_m(:,3);
scan.VelocityNorthMps = episode.state.velocityNed_mps(:,1);
scan.VelocityEastMps = episode.state.velocityNed_mps(:,2);
scan.VelocityDownMps = episode.state.velocityNed_mps(:,3);
scan.RollRad = episode.state.attitudeNedRpy_rad(:,1);
scan.PitchRad = episode.state.attitudeNedRpy_rad(:,2);
scan.YawRad = episode.state.attitudeNedRpy_rad(:,3);
scan.BodyRollRateRadps = episode.state.angularVelocityBody_radps(:,1);
scan.BodyPitchRateRadps = episode.state.angularVelocityBody_radps(:,2);
scan.BodyYawRateRadps = episode.state.angularVelocityBody_radps(:,3);
scan.GoalNorthM = episode.goal.positionNed_m(:,1);
scan.GoalEastM = episode.goal.positionNed_m(:,2);
scan.GoalDownM = episode.goal.positionNed_m(:,3);
scan.DesiredReferenceNorthM = episode.reference.desiredPositionNed_m(:,1);
scan.DesiredReferenceEastM = episode.reference.desiredPositionNed_m(:,2);
scan.DesiredReferenceDownM = episode.reference.desiredPositionNed_m(:,3);
scan.CommandRollRad = episode.action.current(:,1);
scan.CommandPitchRad = episode.action.current(:,2);
scan.CommandYawRateRadps = episode.action.current(:,3);
scan.CommandThrustN = episode.action.current(:,4);
scan.PreviousActionAvailable = episode.action.previousAvailable;
scan.LidarEnabled = episode.lidar.enabled;
scan.ValidReturnFraction = sum(valid,2)/size(valid,2);
scan.NearestValidReturnM = nearest;
scan.DistanceToFinalGoalM = vecnorm(episode.state.positionNed_m ...
    -double(manifest.configuration.mission.waypointsNed(end,:)),2,2);
scan.ObstacleAvoidanceStatus = episode.obstacleAvoidanceStatus;
end

function verifyScanTable(actual,expected)
assert(height(actual)==height(expected) && ...
    isequal(actual.Properties.VariableNames,expected.Properties.VariableNames), ...
    "ARES:RecoveredScanTableMismatch","Scan CSV schema or row count differs.");
for column = 1:width(expected)
    first = actual{:,column}; second = expected{:,column};
    if isnumeric(second) || islogical(second)
        first = double(first); second = double(second);
        assert(isequal(size(first),size(second)) && ...
            isequal(isnan(first),isnan(second)) && ...
            all(abs(first(~isnan(second))-second(~isnan(second)))<=1e-8), ...
            "ARES:RecoveredScanTableMismatch","Scan CSV numeric column differs: %s", ...
            expected.Properties.VariableNames{column});
    else
        assert(isequal(string(first),string(second)), ...
            "ARES:RecoveredScanTableMismatch","Scan CSV identity column differs.");
    end
end
end

function atomicWorkerCheckpoint(checkpointPath,index)
[folder,name] = fileparts(checkpointPath);
suffix = extractAfter(string(name),"checkpoint");
temporary = fullfile(folder,"checkpoint"+suffix+".recovery.tmp.mat");
save(temporary,"index");
movefile(temporary,checkpointPath,"f");
csvPath = fullfile(folder,"index"+suffix+".csv");
temporary = csvPath+".recovery.tmp.csv";
writetable(index,temporary);
movefile(temporary,csvPath,"f");
end

function writeImmutableJson(path,value)
assert(~isfile(path),"ARES:RecoveryRecordAlreadyExists","Recovery record already exists.");
fid = fopen(path,"w","n","UTF-8");
assert(fid~=-1,"ARES:WriteFailed","Cannot create immutable recovery record.");
cleanup = onCleanup(@()fclose(fid));
fprintf(fid,"%s",jsonencode(value,PrettyPrint=true));
end

function hash = fileHash(path)
fid = fopen(path,"rb");
assert(fid~=-1,"ARES:ReadFailed","Cannot read file for recovery provenance: %s",path);
cleanup = onCleanup(@()fclose(fid));
digest = java.security.MessageDigest.getInstance("SHA-256");
while ~feof(fid)
    digest.update(fread(fid,1048576,"*uint8"));
end
hash = lower(string(reshape(dec2hex(typecast(digest.digest(),"uint8"),2).',1,[])));
end

function value = utcNow()
value = string(datetime("now","TimeZone","UTC","Format","yyyy-MM-dd'T'HH:mm:ss.SSSXXX"));
end
