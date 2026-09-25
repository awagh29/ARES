proj = currentProject;
projectRoot = string(proj.RootFolder);

disp("Creating ARES run manifest...");

manifest = ares.data.createRunManifest(projectRoot, "ares_single_uav_top", "nominal-single-uav");

run(fullfile(projectRoot, "scripts", "runSingleUav.m"));

disp("Extracting raw telemetry...");

telemetry = ares.data.extractRawTelemetry(simOut, manifest);

disp("Saving immutable raw run...");

manifestPath = ares.data.saveRuntimeManifest(projectRoot, manifest);

telemetryPath = ares.data.saveRawTelemetry(projectRoot, telemetry);

% Read-back verification.
savedManifest = jsondecode(fileread(manifestPath));
savedData = load(telemetryPath, "telemetry");

assert(string(savedManifest.runId) == savedData.telemetry.runId);

assert(string(savedManifest.schemaVersion) == savedData.telemetry.schemaVersion);

assert(numel(fieldnames(savedData.telemetry.signals)) == 8);

fprintf("\nARES raw run completed successfully.\n");
fprintf("Run ID: %s\n", manifest.runId);
fprintf("Run folder: %s\n", string(fileparts(telemetryPath)));

disp("PASS: Nominal simulation, safety checks and data capture passed.");