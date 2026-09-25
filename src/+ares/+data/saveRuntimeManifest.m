function manifestPath = saveRunManifest(projectRoot, manifest)

arguments
    projectRoot (1,1) string
    manifest (1,1) struct
end

requiredFields = [
    "schemaVersion"
    "runId"
    "createdAtUtc"
    "configuration"
    ];

for fieldName = requiredFields'
    if ~isfield(manifest, fieldName)
        error( ...
            "ARES:InvalidManifest", ...
            "Manifest is missing required field: %s", ...
            fieldName);
    end
end

runFolder = fullfile( ...
    projectRoot, ...
    "data", ...
    "raw", ...
    manifest.runId);

if isfolder(runFolder)
    error( ...
        "ARES:RunAlreadyExists", ...
        "Run directory already exists: %s", ...
        runFolder);
end

mkdir(runFolder);

manifestPath = fullfile(runFolder, "manifest.json");
jsonText = jsonencode(manifest, PrettyPrint=true);

[fileId, errorMessage] = fopen( ...
    manifestPath, ...
    "w", ...
    "n", ...
    "UTF-8");

if fileId == -1
    error( ...
        "ARES:ManifestWriteFailure", ...
        "Could not create manifest: %s", ...
        errorMessage);
end

fileCleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, "%s", jsonText);
clear fileCleanup;

end