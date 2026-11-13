function report = validateSaitsMatlab(modelPath, options)
%VALIDATESAITSMATLAB Verify native inference against exported PyTorch fixtures.
% Returns a report without writing into the model directory. Optionally build
% a MEX in MexDirectory (or a new temporary folder), verify it, and benchmark
% both implementations. Compilation uses MATLAB BLAS and a C++ compiler;
% MATLAB Coder and Deep Learning Toolbox are not required.
arguments
    modelPath (1,1) string
    options.BuildMex (1,1) logical = false
    options.BenchmarkCount (1,1) double {mustBeInteger, mustBePositive} = 100
    options.MexDirectory (1,1) string = ""
    options.AbsoluteTolerance (1,1) double {mustBePositive} = 1e-4
end
assert(isfile(modelPath), 'ARES:MissingSaitsModel', ...
    'Exported SAITS MAT file does not exist: %s', modelPath);
projectRoot = string(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(projectRoot, 'src'));
loaded = load(modelPath, 'model', 'fixtures', 'provenance');
assert(isfield(loaded, 'model') && isfield(loaded, 'fixtures'), ...
    'ARES:InvalidSaitsExport', 'SAITS export must contain model and fixtures.');
model = loaded.model;
fixtures = loaded.fixtures;
assert(strcmp(string(model.schemaVersion), 'ares-saits-native-mat-v1'), ...
    'ARES:UnsupportedSaitsExport', 'Unexpected native SAITS export schema.');
assert(isequal(size(fixtures.rawFeatures, 1), double(model.nSteps)) && ...
    size(fixtures.rawFeatures, 2) == 31 && ...
    isequal(size(fixtures.rawFeatures), size(fixtures.observedMask)), ...
    'ARES:InvalidSaitsFixtures', 'Fixture feature and mask dimensions do not match the model.');
caseCount = size(fixtures.rawFeatures, 3);
assert(caseCount > 0 && size(fixtures.predictionPhysical, 1) == caseCount && ...
    size(fixtures.predictionPhysical, 2) == 14, ...
    'ARES:InvalidSaitsFixtures', 'Fixture predictions require one row per case.');

physicalErrors = zeros(caseCount, 1);
normalizedErrors = zeros(caseCount, 1);
stage1Errors = zeros(caseCount, 1);
stage2Errors = zeros(caseCount, 1);
attentionErrors = zeros(caseCount, 1);
for index = 1:caseCount
    raw = single(fixtures.rawFeatures(:, :, index));
    observed = logical(fixtures.observedMask(:, :, index));
    [physical, normalized, detail] = ares.inference.predictSaits(model, raw, observed);
    assert(all(isfinite(physical)), 'ARES:NonfiniteSaitsPrediction', ...
        'Native SAITS produced a nonfinite prediction.');
    physicalErrors(index) = maximumError(physical, fixtures.predictionPhysical(index, :));
    normalizedErrors(index) = maximumError(normalized, fixtures.predictionNormalized(index, :));
    stage1Errors(index) = maximumError(detail.stage1Normalized, fixtures.stage1Normalized(index, :));
    stage2Errors(index) = maximumError(detail.stage2Normalized, fixtures.stage2Normalized(index, :));
    attentionErrors(index) = maximumError(detail.attentionEndpoint, fixtures.attention(end, :, index));
    % The NaN convenience interface must agree with an explicit mask.
    raw(~observed) = single(nan);
    fromNan = ares.inference.predictSaits(model, raw);
    assert(maximumError(fromNan, physical) <= options.AbsoluteTolerance, ...
        'ARES:SaitsMaskInterfaceMismatch', 'Explicit-mask and NaN interfaces differ.');
end
assert(max(physicalErrors) <= options.AbsoluteTolerance, 'ARES:SaitsParityFailure', ...
    'Native/PyTorch maximum physical prediction error %.9g exceeds %.9g.', ...
    max(physicalErrors), options.AbsoluteTolerance);
assert(max([normalizedErrors; stage1Errors; stage2Errors; attentionErrors]) <= ...
    options.AbsoluteTolerance, 'ARES:SaitsIntermediateParityFailure', ...
    'A native SAITS stage or attention value differs from its PyTorch fixture.');

raw = single(fixtures.rawFeatures(:, :, 1));
observed = logical(fixtures.observedMask(:, :, 1));
ares.inference.predictSaits(model, raw, observed); % Exclude first-call setup.
started = tic;
for repetition = 1:options.BenchmarkCount
    ares.inference.predictSaits(model, raw, observed);
end
nativeSeconds = toc(started) / options.BenchmarkCount;
report.schemaVersion = 'ares-saits-matlab-validation-v1';
report.modelPath = modelPath;
report.modelSha256 = fileSha256(modelPath);
report.predictorSourceSha256 = fileSha256(fullfile(projectRoot, ...
    'src', '+ares', '+inference', 'predictSaits.m'));
report.fixtureCount = caseCount;
report.absoluteTolerance = options.AbsoluteTolerance;
report.maximumPhysicalAbsoluteError = max(physicalErrors);
report.maximumNormalizedAbsoluteError = max(normalizedErrors);
report.maximumStage1AbsoluteError = max(stage1Errors);
report.maximumStage2AbsoluteError = max(stage2Errors);
report.maximumAttentionAbsoluteError = max(attentionErrors);
report.physicalAbsoluteErrorsByCase = physicalErrors;
report.nativeMeanPredictionSeconds = nativeSeconds;
report.benchmarkCount = options.BenchmarkCount;
report.finalSecondStageEndpointOnly = true;
report.allParityChecksPassed = true;
report.matlabRelease = version('-release');
report.platform = computer;
report.mexBuilt = false;

if options.BuildMex
    directory = options.MexDirectory;
    if strlength(directory) == 0
        directory = string(tempname) + "_ares_saits_mex";
    end
    if ~isfolder(directory)
        mkdir(directory);
    end
    mexName = 'saits_endpoint_mex';
    clear(mexName);
    build = buildSaitsMex(directory);
    addpath(directory);
    clear(mexName);
    % The C++ entry always receives the model explicitly; no hidden weight cache.
    includesModel = true;
    mexPhysical = callMex(mexName, includesModel, model, raw, observed);
    mexError = maximumError(mexPhysical, fixtures.predictionPhysical(1, :));
    for index = 2:caseCount
        nextRaw = single(fixtures.rawFeatures(:, :, index));
        nextMask = logical(fixtures.observedMask(:, :, index));
        next = callMex(mexName, includesModel, model, nextRaw, nextMask);
        mexError = max(mexError, maximumError(next, fixtures.predictionPhysical(index, :)));
    end
    assert(mexError <= options.AbsoluteTolerance, 'ARES:SaitsMexParityFailure', ...
        'MEX/PyTorch maximum physical prediction error exceeds the tolerance.');
    started = tic;
    for repetition = 1:options.BenchmarkCount
        callMex(mexName, includesModel, model, raw, observed);
    end
    report.mexMeanPredictionSeconds = toc(started) / options.BenchmarkCount;
    report.maximumMexPhysicalAbsoluteError = mexError;
    report.mexDirectory = directory;
    report.mexFunctionName = mexName;
    report.mexIncludesModelArgument = includesModel;
    report.mexBuilt = true;
    report.mexBackend = build.backend;
    report.mexSourceSha256 = build.sourceSha256;
    report.mexBuilderSourceSha256 = build.builderSourceSha256;
    report.mexCompilerName = build.compilerName;
    report.requiresMatlabCoder = false;
    report.mexFullPath = fullfile(directory, string(mexName) + "." + string(mexext));
    assert(isfile(report.mexFullPath), 'ARES:MissingSaitsMex', ...
        'The generated SAITS MEX binary is missing from its output directory.');
    report.mexSha256 = fileSha256(report.mexFullPath);
    report.mexManifestPath = fullfile(directory, 'saits_endpoint_mex_manifest.json');
    jsonText = jsonencode(report, PrettyPrint=true);
    fileId = fopen(report.mexManifestPath, 'w', 'n', 'UTF-8');
    assert(fileId ~= -1, 'ARES:SaitsMexManifestWriteFailed', ...
        'Could not save the MEX parity/hash manifest.');
    closeFile = onCleanup(@() fclose(fileId));
    fprintf(fileId, '%s\n', jsonText);
    clear closeFile;
end
fprintf('SAITS MATLAB parity passed: %d cases, max physical error %.3g, native %.3f ms/call.\n', ...
    caseCount, report.maximumPhysicalAbsoluteError, 1000 * nativeSeconds);
end

function value = maximumError(actual, expected)
assert(isequal(size(actual), size(expected)), 'ARES:SaitsFixtureShapeMismatch', ...
    'Native and fixture result shapes differ.');
value = max(abs(double(actual(:)) - double(expected(:))));
end

function prediction = callMex(name, includesModel, model, raw, observed)
if includesModel
    prediction = feval(name, model, raw, observed);
else
    prediction = feval(name, raw, observed);
end
end

function digest = fileSha256(path)
fileId = fopen(path, 'rb');
assert(fileId ~= -1, 'ARES:SaitsHashReadFailed', 'Could not read a SAITS artifact for hashing.');
closeFile = onCleanup(@() fclose(fileId));
bytes = fread(fileId, inf, '*uint8');
clear closeFile;
hasher = java.security.MessageDigest.getInstance('SHA-256');
hasher.update(typecast(bytes, 'int8'));
value = typecast(hasher.digest(), 'uint8');
digest = lower(reshape(dec2hex(value, 2).', 1, []));
end
