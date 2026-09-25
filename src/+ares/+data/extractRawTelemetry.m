function telemetry = extractRawTelemetry(simOutput, manifest)
%EXTRACTRAWTELEMETRY Extracts required ARES signals from a simulation.

arguments
    simOutput (1,1) Simulink.SimulationOutput
    manifest (1,1) struct
end

contract = ares.data.defaultDataContract();

if ~isfield(manifest, "runId") || ...
        ~isfield(manifest, "schemaVersion")
    error("ARES:InvalidManifest", ...
        "Manifest requires runId and schemaVersion.");
end

if manifest.schemaVersion ~= contract.schemaVersion
    error("ARES:SchemaMismatch", ...
        "Manifest schema %s does not match contract schema %s.", ...
        manifest.schemaVersion, ...
        contract.schemaVersion);
end

outputNames = string(simOutput.who);

if ~any(outputNames == "logsout")
    error("ARES:MissingLogsout", ...
        "Simulation output does not contain logsout.");
end

logs = simOutput.logsout;
loggedNames = string(logs.getElementNames());

requiredNames = contract.dynamicFields;
requiredNames(requiredNames == "time_s") = [];

missingNames = setdiff(requiredNames, loggedNames);

if ~isempty(missingNames)
    error("ARES:MissingTelemetry", ...
        "Missing required signals: %s", ...
        strjoin(missingNames, ", "));
end

telemetry.schemaVersion = contract.schemaVersion;
telemetry.runId = manifest.runId;
telemetry.coordinateFrame = contract.coordinateFrame;
telemetry.timeBase = contract.timeBase;

telemetry.signals.position_ned_m = extractSignal( ...
    logs, "position_ned_m", 3, "m", ...
    ["north", "east", "down"]);

telemetry.signals.desired_position_ned_m = extractSignal( ...
    logs, "desired_position_ned_m", 3, "m", ...
    ["north", "east", "down"]);

telemetry.signals.velocity_ned_mps = extractSignal( ...
    logs, "velocity_ned_mps", 3, "m/s", ...
    ["north", "east", "down"]);

telemetry.signals.command_roll_rad = extractSignal( ...
    logs, "command_roll_rad", 1, "rad", "roll");

telemetry.signals.command_pitch_rad = extractSignal( ...
    logs, "command_pitch_rad", 1, "rad", "pitch");

telemetry.signals.command_yaw_rate_radps = extractSignal( ...
    logs, "command_yaw_rate_radps", 1, "rad/s", "yaw_rate");

telemetry.signals.command_thrust_n = extractSignal( ...
    logs, "command_thrust_n", 1, "N", "thrust");

telemetry.signals.obstacle_avoidance_status = extractSignal( ...
    logs, "obstacle_avoidance_status", 1, "code", "status");

end

function signal = extractSignal( ...
    logs, signalName, expectedWidth, unitName, componentNames)

element = logs.getElement(signalName);

signal = ares.data.timeseriesToSignal( ...
    element.Values, ...
    expectedWidth);

signal.name = signalName;
signal.unit = unitName;
signal.components = componentNames;

end