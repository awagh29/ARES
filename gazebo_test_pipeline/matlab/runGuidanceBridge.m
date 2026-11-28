function runGuidanceBridge(host, port, projectRoot)
%RUNGUIDANCEBRIDGE Run Atharva's map-temporal guidance for the Gazebo pipeline.
%
% The Python runner (WSL) listens on HOST:PORT; this MATLAB process connects
% and answers one request per 10 Hz guidance tick. It executes, unchanged:
%   ares.mapTemporal.LidarReconstruction  - Atharva's CNN/GRU reconstruction
%   UAV Toolbox Waypoint Follower         - same block settings as the model
%   UAV Toolbox Obstacle Avoidance (VFH)  - same block settings (not in planner mode)
%   ares.mapTemporal.GoalHoldReference    - Atharva's terminal goal hold
%   ares.mapTemporal.SpeedGovernor        - his trajectory-sampling planner +
%                                           braking governor (planner/governor modes)
% exactly as wired in ares.mapTemporal.configureModel/configureGovernor.
% The PID cascade and the vehicle (Gazebo) live on the Python side.
%
% Guidance modes (init message field "guidance"):
%   planner  - EnableReferenceGovernor & EnableLocalPlanner (his default)
%   governor - VFH -> position PID -> braking governor
%   vfh      - original VFH/PID (EnableReferenceGovernor = false)
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
            case "retarget"
                [state, reply] = retarget(state, header);
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
mode = "planner";
if isfield(message, 'guidance') && strlength(string(message.guidance)) > 0
    mode = string(message.guidance);
end
assert(any(mode == ["planner", "governor", "vfh"]), 'ARES:BridgeGuidance', ...
    'guidance must be planner, governor or vfh (got %s).', mode);
o.EnableReferenceGovernor = mode ~= "vfh";
o.EnableLocalPlanner = mode == "planner";
o = ares.mapTemporal.validateOptions(o);
% Export/validate a checkpoint once per MATLAB session.
key = char(o.ModelPath);
if ~isKey(artifacts, key)
    artifacts(key) = ares.mapTemporal.resolveArtifact(projectRoot, o);
end
artifact = artifacts(key);

cfg = ares.config.defaultSingleUav();
% VFH: the compiled copy of the same block (buildVfhMex.m, verified against
% the interpreted object) when it exists, else the interpreted object.
assert(o.VehicleRadiusM == .1 && cfg.avoidance.planningSafetyDistanceM == 1.5, ...
    'ARES:BridgeVfhSettings', 'aresVfhStep hard-codes the baseline radius 0.1 m and safety distance 1.5 m.');
% configureModel widens VFH's safety distance when the governor is enabled.
state.vfhSafety = cfg.avoidance.planningSafetyDistanceM;
if o.EnableReferenceGovernor
    state.vfhSafety = o.MinimumSurfaceClearanceM + o.GovernorMarginM + o.GovernorPlanningBufferM;
end
state.mode = mode;
% The MEX is compiled for the baseline safety distance only.
state.useVfhMex = exist('aresVfhStep_mex', 'file') == 3 && state.vfhSafety == 1.5;
state.vfhReset = true;
state.artifactPath = char(artifact.Path);
state.lookaheadDistance = 5;   % Constant3 in the baseline model
state.lookaheadGain = .5;      % "Lookahead Distance" gain on the VFH direction
state = buildMission(state, o);
validation = artifact.Validation;
reply = struct('ok', true, 'waypoints', state.waypoints, 'artifact_path', string(artifact.Path), ...
    'source_sha256', string(artifact.SourceSha256), 'artifact_sha256', string(artifact.Sha256), ...
    'parity_passed', logical(validation.Passed), 'world_bounds', o.WorldBoundsNedM, ...
    'avoidance_safety_distance', state.vfhSafety, 'guidance', mode, ...
    'cruise_speed_mps', o.CruiseSpeedMps, ...
    'vfh_backend', string(ternary(mode == "planner", "bypassed (planner)", ...
        ternary(state.useVfhMex, "compiled MEX", "interpreted"))));
end

function state = buildMission(state, o)
% (Re)create every goal-dependent object for mission options O, as
% ares.mapTemporal.initializeSimulation/configureModel do for one flight.
planFile = [tempname '.mat'];
options = o; save(planFile, 'options');
% Same follower path as ares.mapTemporal.initializeSimulation.
waypoints = [o.StartState(1:3); o.GoalState(1:3)];
if norm(o.StartState(1:3) - o.GoalState(1:3)) < 1e-6
    inward = mean(o.WorldBoundsNedM, 2)' - o.GoalState(1:3);
    if norm(inward) < 1e-6, inward = [1 0 0]; end
    waypoints(1,:) = o.GoalState(1:3) + .01*inward/norm(inward);
end
% Block settings copied from models/ares_single_uav_top.slx (+ map-temporal edits).
state.follower = uav.sluav.internal.system.WaypointFollower( ...
    'UAVType', 'multirotor', 'StartFrom', 'first', 'TransitionRadiusSource', 'internal', ...
    'TransitionRadius', .1, 'MinLookaheadDistance', .2, 'HoverAtLastWaypoint', true, ...
    'InterpolateYaw', true, 'YawInput', false, 'ShowStatusOutput', true);
state.reconstruction = ares.mapTemporal.LidarReconstruction( ...
    'ArtifactPath', state.artifactPath, 'PlanFile', planFile);
state.goalHold = ares.mapTemporal.GoalHoldReference('PlanFile', planFile);
state.governor = [];
if o.EnableReferenceGovernor
    state.governor = ares.mapTemporal.SpeedGovernor('PlanFile', planFile);
end
if isfield(state, 'planFile') && isfile(state.planFile), delete(state.planFile); end
state.waypoints = waypoints;
state.planFile = planFile;
state.options = o;
end

function [state, reply] = retarget(state, message)
% Swarm hub task: new goal mid-flight. The flight continues from the current
% state, so StartState is the present position/velocity (speed kept inside
% the validated limit). The reconstruction, planner and governor restart for
% the new goal, as for a new mission; VFH (governor/vfh modes) keeps its memory.
assert(isfield(state, 'options'), 'ARES:BridgeNotInitialized', 'Send init first.');
o = state.options;
goal = reshape(double(message.goal), 1, 3);
v = reshape(double(message.velocity), 1, 3);
limit = .95 * o.CruiseSpeedMps;
if norm(v) > limit, v = v * limit / norm(v); end
o.StartState = [reshape(double(message.position), 1, 3) v];
o.StartEulerZYXRad = [double(message.euler_zyx(1)) 0 0];
o.StartBodyRatesRadps = [0 0 0];
o.GoalState = [goal 0 0 0];
o = ares.mapTemporal.validateOptions(o);
state = buildMission(state, o);
reply = struct('ok', true, 'goal', goal, 'waypoints', state.waypoints);
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
direction = NaN(3, 1); avoidanceStatus = NaN;
if state.mode == "planner"
    % configureModel rewires the avoidance switch to the follower and comments VFH out.
    nominal = single(lookahead(:)); nominalYaw = single(followerYaw);
else
    vfhArgs = {state.vfhReset, single(p), single(quaternion(:)), single(cloud), single(lookahead(:))};
    if state.useVfhMex
        [direction, avoidanceYaw, avoidanceStatus] = aresVfhStep_mex(vfhArgs{:});
    else
        [direction, avoidanceYaw, avoidanceStatus] = aresVfhStep(vfhArgs{:}, state.vfhSafety);
    end
    nominal = single(p) + single(state.lookaheadGain) * single(direction(:));
    nominalYaw = single(avoidanceYaw);
end
state.vfhReset = false;   % fresh VFH memory at the start of every run only
avoidanceS = toc(mark);
% Map-temporal always uses the selected cloud: measured scans when healthy,
% the learned reconstruction during outages.
[reference, yawReference, holding] = state.goalHold(nominal, nominalYaw, single(p), single(euler), cloud);
reply = struct('ok', true, 'desired_position', double(reference(:))', ...
    'desired_yaw', double(yawReference), 'reconstruction_used', logical(used), ...
    'prediction_valid', logical(valid), 'inference_s', double(inference), ...
    'terminal', logical(terminal), 'holding', logical(holding), ...
    'lookahead', double(lookahead(:))', 'follower_yaw', double(followerYaw), ...
    'follower_status', double(followerStatus), 'avoidance_status', double(avoidanceStatus), ...
    'vfh_direction', double(direction(:))', 'history_steps', packet(15), ...
    'blackout_age_s', packet(16), 'prior_min_range_m', packet(21));
if ~isempty(state.governor)
    % Position-PID velocity request (from Python); replaced by the planner in planner mode.
    request = zeros(3, 1);
    if isfield(message, 'velocity_request'), request = reshape(double(message.velocity_request), 3, 1); end
    mark = tic;
    [command, governorPacket, plannerPacket] = state.governor(request, p, v, euler, cloud, logical(holding));
    reply.governor_s = toc(mark);
    reply.velocity_command = double(command(:))';
    reply.governor = double(governorPacket(:))';   % configureModel GovernorPacketOrder
    reply.planner = double(plannerPacket(:))';     % configureModel PlannerPacketOrder
end
reply.bridge_s = toc(timer);
reply.reconstruction_block_s = reconstructionS;
reply.avoidance_s = avoidanceS;
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
