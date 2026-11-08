function telemetryPath = saveRawTelemetry(projectRoot, telemetry, dataSubdirectory)

arguments
    projectRoot (1,1) string
    telemetry (1,1) struct
    dataSubdirectory (1,1) string = "raw"
end

validateDataSubdirectory(dataSubdirectory);

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

runFolder = fullfile(projectRoot, ...
    "data", ...
    dataSubdirectory, ...
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

filename = savedManifest.simulation.scenarioName + "_telemetry" + ".mat";
telemetryPath = fullfile(runFolder, filename);

if isfile(telemetryPath)
    error("ARES:TelemetryAlreadyExists", ...
        "Telemetry already exists and will not be overwritten: %s", ...
        telemetryPath);
end

save(telemetryPath, "telemetry", "-v7.3");

end

function validateDataSubdirectory(value)
valid = ~isempty(regexp(value,"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$","once"));
reserved = ~isempty(regexpi(value,"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$","once"));
assert(valid && ~reserved,"ARES:InvalidDataSubdirectory", ...
    "Data subdirectory must be one safe folder name using letters, digits, underscores or hyphens.");
end
