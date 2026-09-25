function telemetryPath = saveRawTelemetry(projectRoot, telemetry)
%SAVERAWTELEMETRY Saves telemetry beside its matching run manifest.

arguments
    projectRoot (1,1) string
    telemetry (1,1) struct
end

requiredFields = [
    "schemaVersion"
    "runId"
    "signals"
    ];

for fieldName = requiredFields'
    if ~isfield(telemetry, fieldName)
        error("ARES:InvalidTelemetry", ...
            "Telemetry is missing required field: %s", ...
            fieldName);
    end
end

runFolder = fullfile( ...
    projectRoot, ...
    "data", ...
    "raw", ...
    telemetry.runId);

manifestPath = fullfile(runFolder, "manifest.json");

if ~isfile(manifestPath)
    error("ARES:MissingManifest", ...
        "The matching manifest must be saved first: %s", ...
        manifestPath);
end

savedManifest = jsondecode(fileread(manifestPath));

if string(savedManifest.runId) ~= telemetry.runId
    error("ARES:RunIdMismatch", ...
        "Manifest and telemetry run IDs do not match.");
end

if string(savedManifest.schemaVersion) ~= ...
        telemetry.schemaVersion
    error("ARES:SchemaMismatch", ...
        "Manifest and telemetry schema versions do not match.");
end

telemetryPath = fullfile(runFolder, "telemetry.mat");

if isfile(telemetryPath)
    error("ARES:TelemetryAlreadyExists", ...
        "Telemetry already exists and will not be overwritten: %s", ...
        telemetryPath);
end

save(telemetryPath, "telemetry", "-v7.3");

end