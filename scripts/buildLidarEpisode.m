function episode = buildLidarEpisode(telemetryPath, outputPath)
%BUILDLIDAREPISODE Create a model-neutral, healthy-LiDAR episode.
% Raw telemetry remains the source of truth. Continuous state is linearly
% interpolated to scan timestamps; commands and goals use zero-order hold.
% Duplicate native timestamps retain the last value. No extrapolation is used.

arguments
    telemetryPath (1,1) string
    outputPath (1,1) string = ""
end

assert(isfile(telemetryPath), "ARES:MissingTelemetryFile", ...
    "Telemetry file does not exist: %s", telemetryPath);
loaded = load(telemetryPath, "telemetry");
assert(isfield(loaded, "telemetry") && isstruct(loaded.telemetry), ...
    "ARES:InvalidTelemetryFile", "MAT file must contain telemetry.");
telemetry = loaded.telemetry;
assert(isfield(telemetry, "signals") && isfield(telemetry, "runId") && ...
    isfield(telemetry,"schemaVersion") && isfield(telemetry,"timeBase") && ...
    string(telemetry.timeBase)=="simulation-time" && ...
    strlength(string(telemetry.runId)) > 0, ...
    "ARES:InvalidTelemetryFile", "Telemetry requires signals and runId.");
assert(isfield(telemetry, "coordinateFrame") && ...
    string(telemetry.coordinateFrame) == "NED", ...
    "ARES:UnexpectedCoordinateFrame", "Expected telemetry in NED.");
assert(isfield(telemetry, "lidarMetadata"), ...
    "ARES:MissingLidarMetadata", "Telemetry requires lidarMetadata.");

signals = telemetry.signals;
[time_s, ranges] = readSignal(required(signals, "nominal_observation"), ...
    "nominal_observation");
assert(numel(time_s) >= 2 && all(diff(time_s) > 0), ...
    "ARES:InvalidLidarTimeline", ...
    "LiDAR needs at least two strictly increasing scan timestamps.");
scanCount = numel(time_s);
beamCount = size(ranges, 2);
metadata = telemetry.lidarMetadata;
assert(isfield(metadata, "gridSize") && ...
    prod(double(metadata.gridSize)) == beamCount, ...
    "ARES:LidarGridMismatch", "LiDAR grid does not match observation width.");
assert(isfield(metadata, "maximumRangeM") && ...
    isscalar(metadata.maximumRangeM) && ...
    isfinite(metadata.maximumRangeM) && metadata.maximumRangeM > 0, ...
    "ARES:InvalidMaximumLidarRange", "Maximum LiDAR range must be positive.");
assert(all(ranges >= 0 & ranges <= metadata.maximumRangeM, "all"), ...
    "ARES:InvalidLidarRanges", "LiDAR ranges exceed their recorded limits.");

validValues = aligned(required(signals, "nominal_lidar_valid_return_mask"), ...
    time_s, beamCount, "nominal_lidar_valid_return_mask");
assert(all(validValues == 0 | validValues == 1, "all"), ...
    "ARES:InvalidLidarValidityMask", "Validity mask must contain zero or one.");
validMask = logical(validValues);
assert(all(ranges(validMask) > 0), "ARES:InvalidValidLidarRange", ...
    "A valid LiDAR return must have positive range.");
assert(all(ranges(~validMask) == metadata.maximumRangeM), ...
    "ARES:InvalidNoReturnEncoding", ...
    "Absent LiDAR returns must use maximum range with a false validity mask.");

if isfield(signals, "nominal_lidar_ranges_m")
    recordedRanges = aligned(signals.nominal_lidar_ranges_m, ...
        time_s, beamCount, "nominal_lidar_ranges_m");
    assert(isequal(recordedRanges, ranges), "ARES:LidarRangeMismatch", ...
        "Nominal observation differs from nominal LiDAR ranges.");
end

% Check every native enable sample, not only the values sampled at scan times.
[~, nativeEnabled] = readSignal(required(signals, "lidar_enabled"), ...
    "lidar_enabled", 1);
assert(all(nativeEnabled == 1), "ARES:UnexpectedLidarOutage", ...
    "A healthy episode must keep LiDAR enabled at every native sample.");
enabledAtScan = sample(signals.lidar_enabled, time_s, "previous", ...
    "lidar_enabled", 1);
assert(all(enabledAtScan == 1), "ARES:UnexpectedLidarOutage", ...
    "LiDAR is not enabled at every scan.");
checkHealthyMask(signals, "observation_availability_mask", ...
    time_s, beamCount, 1);
checkHealthyMask(signals, "missing_observation_mask", time_s, beamCount, 0);
checkHealthyMask(signals, "lidar_failure_active", time_s, 1, 0);

episode.schemaVersion = "lidar-episode-v1";
episode.runId = string(telemetry.runId);
episode.sourceTelemetryPath = telemetryPath;
episode.sourceTelemetrySchemaVersion = string(telemetry.schemaVersion);
episode.coordinateFrame = "NED";
episode.timeBase = string(telemetry.timeBase);
episode.time_s = time_s;
episode.sampleIndex = uint32((1:scanCount)');
episode.lidar.ranges_m = ranges;
episode.lidar.validReturnMask = validMask;
episode.lidar.availabilityMask = true(scanCount, beamCount);
episode.lidar.missingMask = false(scanCount, beamCount);
episode.lidar.enabled = logical(enabledAtScan);

episode.state.positionNed_m = sample(required(signals, "position_ned_m"), ...
    time_s, "linear", "position_ned_m", 3);
episode.state.velocityNed_mps = sample(required(signals, "velocity_ned_mps"), ...
    time_s, "linear", "velocity_ned_mps", 3);

if isfield(signals, "attitude_ned_rpy_rad")
    % Unwrap angular components before interpolation, then wrap to [-pi, pi].
    episode.state.attitudeNedRpy_rad = sample( ...
        signals.attitude_ned_rpy_rad, time_s, "angular", ...
        "attitude_ned_rpy_rad", 3);
end
if isfield(signals, "angular_velocity_body_radps")
    episode.state.angularVelocityBody_radps = sample( ...
        signals.angular_velocity_body_radps, time_s, "linear", ...
        "angular_velocity_body_radps", 3);
end

episode.reference.desiredPositionNed_m = sample( ...
    required(signals, "desired_position_ned_m"), time_s, "previous", ...
    "desired_position_ned_m", 3);
episode.reference.source = "avoidance-adjusted controller reference";
if isfield(telemetry,"mission") && isfield(telemetry.mission,"finalGoalPositionNed_m")
    finalGoal=double(telemetry.mission.finalGoalPositionNed_m(:)');
    assert(isequal(size(finalGoal),[1 3]) && all(isfinite(finalGoal)), ...
        "ARES:InvalidMissionGoal","Final mission goal must contain three finite NED coordinates.");
    episode.goal.positionNed_m=repmat(finalGoal,scanCount,1);
    episode.goal.source="fixed final mission goal";
else
    % Older telemetry did not distinguish the mission goal from the reference.
    % Keep this explicit rather than guessing its final waypoint.
    episode.goal.positionNed_m=episode.reference.desiredPositionNed_m;
    episode.goal.source="controller-local-reference; final mission goal unavailable";
end
actionSignals = ["command_roll_rad", "command_pitch_rad", ...
    "command_yaw_rate_radps", "command_thrust_n"];
episode.action.current = zeros(scanCount, numel(actionSignals));
for index = 1:numel(actionSignals)
    name = actionSignals(index);
    episode.action.current(:, index) = sample(required(signals, name), ...
        time_s, "previous", name, 1);
end
episode.action.components = actionSignals;
episode.action.units = ["rad", "rad", "rad/s", "N"];
% There is no recorded predecessor to the first scan. NaN plus an explicit
% availability flag avoids inventing a zero controller action at that scan.
episode.action.previous = [nan(1, 4); episode.action.current(1:end-1, :)];
episode.action.previousAvailable = [false; true(scanCount-1, 1)];

if isfield(signals, "obstacle_avoidance_status")
    episode.obstacleAvoidanceStatus = sample( ...
        signals.obstacle_avoidance_status, time_s, "previous", ...
        "obstacle_avoidance_status", 1);
end

episode.metadata = metadata;
episode.metadata.numberOfScans = scanCount;
episode.metadata.observationDimension = beamCount;
episode.metadata.nominalScanPeriod_s = median(diff(time_s));
episode.metadata.validReturnFraction = nnz(validMask) / numel(validMask);
episode.metadata.stateInterpolation = "linear; angles unwrapped then wrapped";
episode.metadata.goalAndActionInterpolation = "previous";
episode.metadata.duplicateNativeTimestampPolicy = "last";
episode.metadata.extrapolationUsed = false;
episode.metadata.timestampBoundaryTolerance_s = timestampBoundaryTolerance();
episode.metadata.timestampBoundaryPolicy = ...
    "clip query endpoints within tolerance to native bounds; preserve LiDAR timestamps";
episode.metadata.previousActionDefinition = "action at previous LiDAR scan";
episode.metadata.firstPreviousActionUnavailable = true;
episode.metadata.condition = "healthy-lidar-enabled";

outputPath = resolveOutputPath(telemetryPath, outputPath);
assert(~isfile(outputPath), "ARES:EpisodeAlreadyExists", ...
    "Episode already exists and will not be overwritten: %s", outputPath);
[folder, ~, ~] = fileparts(outputPath);
if strlength(folder) > 0 && ~isfolder(folder)
    mkdir(folder);
end
save(outputPath, "episode", "-v7.3");
end

function signal = required(signals, name)
assert(isfield(signals, name), "ARES:MissingTelemetrySignal", ...
    "Telemetry signal is missing: %s", name);
signal = signals.(name);
end

function [time_s, values] = readSignal(signal, name, expectedWidth)
if nargin < 3
    expectedWidth = [];
end
if isa(signal, "timeseries")
    time_s = double(signal.Time(:));
    raw = signal.Data;
    if signal.IsTimeFirst
        values = reshape(raw, numel(time_s), []);
    else
        values = reshape(raw, [], numel(time_s)).';
    end
else
    assert(isstruct(signal) && isfield(signal, "time_s") && ...
        isfield(signal, "values"), "ARES:InvalidTelemetrySignal", ...
        "Signal %s requires canonical time_s and values fields.", name);
    time_s = double(signal.time_s(:));
    values = signal.values;
end
assert(~isempty(time_s) && all(isfinite(time_s)) && ...
    isreal(time_s) && all(diff(time_s) >= 0), ...
    "ARES:InvalidSignalTime", ...
    "Signal %s has empty, nonfinite or decreasing timestamps.", name);
assert((isnumeric(values) || islogical(values)) && isreal(values) && ...
    ismatrix(values) && size(values, 1) == numel(time_s) && ...
    size(values, 2) > 0 && all(isfinite(double(values)), "all"), ...
    "ARES:InvalidSignalValues", ...
    "Signal %s requires finite samples in rows aligned with timestamps.", name);
assert(isempty(expectedWidth) || size(values, 2) == expectedWidth, ...
    "ARES:UnexpectedSignalWidth", "Signal %s has an unexpected width.", name);
end

function values = aligned(signal, masterTime, width, name)
[nativeTime, values] = readSignal(signal, name, width);
assert(isequal(nativeTime, masterTime), "ARES:AlignedSignalTimeMismatch", ...
    "Signal %s does not have exactly the LiDAR timestamps.", name);
end

function values = sample(signal, masterTime, method, name, width)
[nativeTime, nativeValues] = readSignal(signal, name, width);
[nativeTime, lastIndex] = unique(nativeTime, "last");
nativeValues = double(nativeValues(lastIndex, :));
boundaryTolerance_s = timestampBoundaryTolerance();
assert(nativeTime(1) - masterTime(1) <= boundaryTolerance_s && ...
    masterTime(end) - nativeTime(end) <= boundaryTolerance_s, ...
    "ARES:ResamplingOutsideTimeRange", ...
    "Signal %s does not cover the full LiDAR timeline.", name);
% Solver and sensor timestamps can represent the same endpoint a few ULPs
% apart. Clip only queries already accepted by the coverage tolerance. This
% preserves the recorded scan times and never asks interp1 to extrapolate.
queryTime = min(max(masterTime, nativeTime(1)), nativeTime(end));
if numel(nativeTime) == 1
    assert(all(queryTime == nativeTime), "ARES:InsufficientSignalSamples", ...
        "Signal %s cannot cover multiple scan times.", name);
    values = repmat(nativeValues, numel(masterTime), 1);
elseif method == "angular"
    values = interp1(nativeTime, unwrap(nativeValues, [], 1), queryTime, "linear");
    values = atan2(sin(values), cos(values));
else
    values = interp1(nativeTime, nativeValues, queryTime, method);
end
assert(all(isfinite(values), "all"), "ARES:NonfiniteResampledSignal", ...
    "Resampling %s produced nonfinite values.", name);
end

function tolerance_s = timestampBoundaryTolerance()
tolerance_s = 1e-9;
end

function checkHealthyMask(signals, name, time_s, width, expected)
if isfield(signals, name)
    values = aligned(signals.(name), time_s, width, name);
    assert(all(values == expected, "all"), "ARES:UnexpectedLidarOutage", ...
        "Signal %s contradicts a healthy LiDAR episode.", name);
end
end

function outputPath = resolveOutputPath(telemetryPath, outputPath)
[sourceFolder, stem] = fileparts(telemetryPath);
filename = stem + "_lidar_episode.mat";
if strlength(outputPath) == 0
    outputPath = fullfile(sourceFolder, filename);
elseif isfolder(outputPath)
    outputPath = fullfile(outputPath, filename);
else
    [~, ~, extension] = fileparts(outputPath);
    if strlength(extension) == 0
        outputPath = fullfile(outputPath, filename);
    else
        assert(strcmpi(extension, ".mat"), "ARES:InvalidEpisodeExtension", ...
            "Episode output must be a .mat file or a directory.");
    end
end
assert(outputPath ~= telemetryPath, "ARES:EpisodeWouldReplaceTelemetry", ...
    "Episode output must differ from its raw telemetry source.");
end
