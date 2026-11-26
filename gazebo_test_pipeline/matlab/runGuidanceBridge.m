function runGuidanceBridge(host, port, projectRoot)
%RUNGUIDANCEBRIDGE Run Atharva's map-temporal guidance for the Gazebo pipeline.
%
% The Python runner (WSL) listens on HOST:PORT; this MATLAB process connects
% and answers one request per 10 Hz guidance tick. It executes, unchanged:
%   ares.mapTemporal.LidarReconstruction  - Atharva's CNN/GRU reconstruction
%   UAV Toolbox Waypoint Follower         - same block settings as the model
%   UAV Toolbox Obstacle Avoidance (VFH)  - same block settings as the model
%   ares.mapTemporal.GoalHoldReference    - Atharva's terminal goal hold
% exactly as wired in ares.mapTemporal.configureModel. The PID cascade and
% the vehicle (Gazebo) live on the Python side.
%
% Messages (little-endian): uint32 headerBytes, JSON header, uint32
% payloadBytes, payload. Step payload = float32 N-by-3 sensor-frame cloud
% (row-major), sent only while the lidar is available.
arguments
    host (1,1) string = "localhost"
    port (1,1) double = 50555
    projectRoot (1,1) string = string(fileparts(fileparts(fileparts(mfilename('fullpath')))))
end
addpath(fullfile(projectRoot,'src'));
client = tcpclient(host, port, "Timeout", 3600, "ConnectTimeout", 120);
cleanup = onCleanup(@() clear('client'));
fprintf('ARES guidance bridge connected to %s:%d\n', host, port);
artifacts = containers.Map();
state = struct();
while true
    [header, payload] = receive(client);
    try
        switch string(header.type)
            case "init"
                [state, reply] = initialize(header, projectRoot, artifacts);
            case "step"
                [state, reply] = stepGuidance(state, header, payload);
            case "end"
                state = struct(); reply = struct('ok', true);
            case "quit"
                send(client, struct('ok', true)); return;
            otherwise
                error('ARES:BridgeMessage', 'Unknown message type %s', header.type);
        end
    catch exception
        reply = struct('ok', false, 'error', string(exception.identifier) + ": " + exception.message);
    end
    send(client, reply);
end
end

function [state, reply] = initialize(message, projectRoot, artifacts)
o = ares.mapTemporal.defaultOptions();
names = fieldnames(message.options);
for k = 1:numel(names)
    value = message.options.(names{k});
    if iscell(value), value = cell2mat(value); end
    if isnumeric(value) && isvector(value) && ~isscalar(value)
        value = reshape(value, 1, []);   % jsondecode returns columns; options use rows
    elseif ischar(value)
        value = string(value);
    end
    o.(names{k}) = value;
end
o.OutageIntervalsS = reshape(double(o.OutageIntervalsS), [], 2);
o.ModelPath = string(message.model_path);
if isfield(message, 'export_python') && strlength(string(message.export_python)) > 0
    o.PythonExecutable = string(message.export_python);
end
o.ShowSimulation = false;
o = ares.mapTemporal.validateOptions(o);
% Export/validate a checkpoint once per MATLAB session.
key = char(o.ModelPath);
if ~isKey(artifacts, key)
    artifacts(key) = ares.mapTemporal.resolveArtifact(projectRoot, o);
end
artifact = artifacts(key);

planFile = [tempname '.mat'];
options = o; save(planFile, 'options');
% Same follower path as ares.mapTemporal.initializeSimulation.
waypoints = [o.StartState(1:3); o.GoalState(1:3)];
if norm(o.StartState(1:3) - o.GoalState(1:3)) < 1e-6
    inward = mean(o.WorldBoundsNedM, 2)' - o.GoalState(1:3);
    if norm(inward) < 1e-6, inward = [1 0 0]; end
    waypoints(1,:) = o.GoalState(1:3) + .01*inward/norm(inward);
end
cfg = ares.config.defaultSingleUav();
% Block settings copied from models/ares_single_uav_top.slx (+ map-temporal edits).
state.follower = uav.sluav.internal.system.WaypointFollower( ...
    'UAVType', 'multirotor', 'StartFrom', 'first', 'TransitionRadiusSource', 'internal', ...
    'TransitionRadius', .1, 'MinLookaheadDistance', .2, 'HoverAtLastWaypoint', true, ...
    'InterpolateYaw', true, 'YawInput', false, 'ShowStatusOutput', true);
% VFH: the compiled copy of the same block (buildVfhMex.m, verified against
% the interpreted object) when it exists, else the interpreted object.
assert(o.VehicleRadiusM == .1 && cfg.avoidance.planningSafetyDistanceM == 1.5, ...
    'ARES:BridgeVfhSettings', 'aresVfhStep hard-codes the baseline radius 0.1 m and safety distance 1.5 m.');
state.useVfhMex = exist('aresVfhStep_mex', 'file') == 3;
state.vfhReset = true;
state.reconstruction = ares.mapTemporal.LidarReconstruction( ...
    'ArtifactPath', char(artifact.Path), 'PlanFile', planFile);
state.goalHold = ares.mapTemporal.GoalHoldReference('PlanFile', planFile);
state.waypoints = waypoints;
state.lookaheadDistance = 5;   % Constant3 in the baseline model
state.lookaheadGain = .5;      % "Lookahead Distance" gain on the VFH direction
state.planFile = planFile;
validation = artifact.Validation;
reply = struct('ok', true, 'waypoints', waypoints, 'artifact_path', string(artifact.Path), ...
    'source_sha256', string(artifact.SourceSha256), 'artifact_sha256', string(artifact.Sha256), ...
    'parity_passed', logical(validation.Passed), 'world_bounds', o.WorldBoundsNedM, ...
    'avoidance_safety_distance', cfg.avoidance.planningSafetyDistanceM, ...
    'vfh_backend', string(ternary(state.useVfhMex, "compiled MEX", "interpreted")));
end

function [state, reply] = stepGuidance(state, message, payload)
assert(isfield(state, 'reconstruction'), 'ARES:BridgeNotInitialized', 'Send init first.');
t = double(message.time);
p = reshape(double(message.position), 3, 1);
v = reshape(double(message.velocity), 3, 1);
euler = reshape(double(message.euler_zyx), 3, 1);   % [yaw pitch roll]
rates = reshape(double(message.body_rates), 3, 1);
available = logical(message.available);
if available
    points = reshape(typecast(payload, 'single'), 3, [])';
    assert(isequal(size(points), [10740 3]), 'ARES:BridgeCloud', 'Expected a 10740x3 organized cloud.');
else
    points = NaN(10740, 3, 'single');   % never inspected while unavailable
end
timer = tic;
[cloud, used, ~, valid, ~, inference, ~, ~, ~, ~, ~, ~, ~, ~, terminal, packet] = ...
    state.reconstruction(points, single(p), single(v), single(euler), single(rates), double(available), t);
reconstructionS = toc(timer);
pose = [p; euler(1)];
[lookahead, ~, followerYaw, ~, followerStatus] = state.follower(pose, state.waypoints, state.lookaheadDistance);
quaternion = eul2quat(euler', 'ZYX');
mark = tic;
vfhArgs = {state.vfhReset, single(p), single(quaternion(:)), single(cloud), single(lookahead(:))};
if state.useVfhMex
    [direction, avoidanceYaw, avoidanceStatus] = aresVfhStep_mex(vfhArgs{:});
else
    [direction, avoidanceYaw, avoidanceStatus] = aresVfhStep(vfhArgs{:});
end
state.vfhReset = false;   % fresh VFH memory at the start of every run only
avoidanceS = toc(mark);
% Map-temporal always selects VFH: measured scans when healthy, the learned
% reconstruction during outages (schedule OR reconstruction-used).
desired = single(p) + single(state.lookaheadGain) * single(direction(:));
[reference, yawReference, holding] = state.goalHold(desired, single(avoidanceYaw), single(p), single(euler));
reply = struct('ok', true, 'desired_position', double(reference(:))', ...
    'desired_yaw', double(yawReference), 'reconstruction_used', logical(used), ...
    'prediction_valid', logical(valid), 'inference_s', double(inference), ...
    'terminal', logical(terminal), 'holding', logical(holding), ...
    'lookahead', double(lookahead(:))', 'follower_yaw', double(followerYaw), ...
    'follower_status', double(followerStatus), 'avoidance_status', double(avoidanceStatus), ...
    'vfh_direction', double(direction(:))', 'history_steps', packet(15), ...
    'blackout_age_s', packet(16), 'prior_min_range_m', packet(21), ...
    'bridge_s', toc(timer), 'reconstruction_block_s', reconstructionS, 'avoidance_s', avoidanceS);
end

function value = ternary(condition, whenTrue, whenFalse)
if condition, value = whenTrue; else, value = whenFalse; end
end

function [header, payload] = receive(client)
headerBytes = double(read(client, 1, "uint32"));
header = jsondecode(native2unicode(read(client, headerBytes, "uint8"), 'UTF-8'));
payloadBytes = double(read(client, 1, "uint32"));
payload = uint8([]);
if payloadBytes > 0, payload = read(client, payloadBytes, "uint8"); end
end

function send(client, reply)
bytes = unicode2native(jsonencode(reply), 'UTF-8');
write(client, uint32(numel(bytes)));
write(client, uint8(bytes));
write(client, uint32(0));
end
