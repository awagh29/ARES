function manifest = createRunManifest(projectRoot, modelName, scenarioName)

arguments
    projectRoot (1,1) string
    modelName (1,1) string = "ares_single_uav_top"
    scenarioName (1,1) string = "nominal-single-uav"
end

contract = ares.data.defaultDataContract();
cfg = ares.config.defaultSingleUav();

createdAt = datetime( ...
    "now", "TimeZone", "UTC", "Format", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX");

manifest.schemaVersion = contract.schemaVersion;
manifest.runId = string(java.util.UUID.randomUUID());
manifest.createdAtUtc = string(createdAt);

manifest.system.name = contract.systemName;
manifest.system.vehicleScope = contract.vehicleScope;

manifest.simulation.modelName = modelName;
manifest.simulation.scenarioName = scenarioName;
manifest.simulation.coordinateFrame = contract.coordinateFrame;
manifest.simulation.timeBase = contract.timeBase;

manifest.software.matlabRelease = string(version("-release"));
manifest.software.platform = string(computer);

[commitStatus, commitOutput] = system( ...
    "git -C """ + projectRoot + """ rev-parse HEAD");

if commitStatus == 0
    manifest.repository.commit = strtrim(string(commitOutput));
else
    manifest.repository.commit = "unavailable";
end

[gitStatusCode, gitOutput] = system( ...
    "git -C """ + projectRoot + """ status --porcelain");

manifest.repository.isDirty = ...
    gitStatusCode ~= 0 || strlength(strtrim(string(gitOutput))) > 0;

manifest.configuration = cfg;

end