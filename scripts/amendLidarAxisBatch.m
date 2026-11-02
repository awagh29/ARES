function amendment = amendLidarAxisBatch(batchPath, projectRoot, reason)
%AMENDLIDARAXISBATCH Append a traceable collection-code hotfix to a batch.
% The original plan/metadata and completed run artifacts remain unchanged.
% Only captureLidarAxisBatch and buildLidarEpisode may change. Every simulator,
% initializer, configuration, data helper, and software version stays frozen.
arguments
    batchPath (1,1) string
    projectRoot (1,1) string
    reason (1,1) string
end
reason = strip(reason);
assert(strlength(reason)>0, "ARES:AmendmentReasonRequired", ...
    "Provide a concrete explanation of the collection-code correction.");
assert(isfolder(batchPath) && isfile(fullfile(batchPath,"plan.mat")), ...
    "ARES:InvalidBatch", "Batch plan was not found: %s", batchPath);
assert(isfile(fullfile(projectRoot,"ares.prj")), ...
    "ARES:InvalidProject", "ARES project not found: %s", projectRoot);
addpath(fullfile(projectRoot,"src"),fullfile(projectRoot,"scripts"),"-end");
saved = load(fullfile(batchPath,"plan.mat"),"metadata");
assert(isfield(saved,"metadata") && isfield(saved.metadata,"batchId") && ...
    isfield(saved.metadata,"sourceSnapshot"), ...
    "ARES:InvalidBatch", "The batch lacks immutable source metadata.");
metadata = saved.metadata;
expected = metadata.sourceSnapshot;

% Require a contiguous immutable chain rooted in the original metadata.
entries = dir(fullfile(batchPath,"source_amendment_*.mat"));
numbers = zeros(numel(entries),1);
for k=1:numel(entries)
    token = regexp(entries(k).name,"^source_amendment_([0-9]+)\.mat$", ...
        "tokens","once");
    assert(~isempty(token),"ARES:InvalidAmendmentChain", ...
        "Unexpected amendment filename: %s",entries(k).name);
    numbers(k) = str2double(token{1});
end
[numbers,order] = sort(numbers);
entries = entries(order);
assert(isequal(numbers,(1:numel(entries))'), ...
    "ARES:InvalidAmendmentChain", "Amendment sequence must be contiguous from 001.");
for k=1:numel(entries)
    record = load(fullfile(batchPath,entries(k).name),"amendment");
    assert(isfield(record,"amendment"),"ARES:InvalidAmendmentChain", ...
        "Amendment record is missing: %s",entries(k).name);
    prior = record.amendment;
    assert(all(isfield(prior,["batchId","fromSnapshot","toSnapshot", ...
        "reason","createdAtUtc","changedFunctions"])), ...
        "ARES:InvalidAmendmentChain", "Amendment record lacks required fields.");
    assert(string(prior.batchId)==string(metadata.batchId) && ...
        isequal(prior.fromSnapshot,expected), ...
        "ARES:InvalidAmendmentChain", ...
        "Amendment %d does not extend this batch's frozen source snapshot.",k);
    verifiedChanges = allowedChanges(prior.fromSnapshot,prior.toSnapshot);
    assert(isequal(verifiedChanges,prior.changedFunctions), ...
        "ARES:InvalidAmendmentChain", ...
        "Amendment %d has an inconsistent changed-function record.",k);
    expected = prior.toSnapshot;
end

% Preserve the snapshot's exact shape/order while recomputing every value.
actual = expected;
actual.matlabVersion = string(version);
actual.platform = string(computer);
actual.toolboxes = ver;
for k=1:numel(actual.files)
    actual.files(k).sha256 = fileHash(fullfile(projectRoot, ...
        string(actual.files(k).path)));
end
for k=1:numel(actual.collectionCode)
    name = string(actual.collectionCode(k).name);
    resolved = string(which(name));
    assert(strlength(resolved)>0,"ARES:MissingCollectionFunction", ...
        "Collection function cannot be resolved: %s",name);
    actual.collectionCode(k).sha256 = fileHash(resolved);
end
changedFunctions = allowedChanges(expected,actual);
assert(~isempty(changedFunctions),"ARES:AmendmentHasNoChanges", ...
    "The current collection code already matches the latest source snapshot.");

amendment.schemaVersion = "lidar-axis-source-amendment-v1";
amendment.batchId = metadata.batchId;
amendment.sequence = numel(entries)+1;
amendment.createdAtUtc = string(datetime("now","TimeZone","UTC", ...
    "Format","yyyy-MM-dd'T'HH:mm:ss.SSSXXX"));
amendment.reason = reason;
amendment.fromSnapshot = expected;
amendment.toSnapshot = actual;
amendment.changedFunctions = changedFunctions;
amendment.provenanceNote = "Original plan.mat and run provenance remain unchanged; " + ...
    "this record authorizes only subsequent collection/episode attempts under the listed hotfix.";
stem = fullfile(batchPath,sprintf("source_amendment_%03d",amendment.sequence));
matPath = stem+".mat";
jsonPath = stem+".json";
assert(~isfile(matPath) && ~isfile(jsonPath),"ARES:AmendmentAlreadyExists", ...
    "An amendment artifact already exists and will not be overwritten: %s",stem);

% Writing JSON first leaves the MAT chain untouched if descriptive export fails.
% Existing amendment files are never reused or overwritten.
writeJsonExclusive(jsonPath,amendment);
assert(~isfile(matPath),"ARES:AmendmentAlreadyExists", ...
    "An amendment MAT already exists and will not be overwritten: %s",matPath);
save(matPath,"amendment");
readback = load(matPath,"amendment");
assert(isequaln(readback.amendment,amendment),"ARES:AmendmentReadbackFailed", ...
    "Amendment MAT failed read-back verification: %s",matPath);
fprintf("SOURCE_AMENDMENT %d batch=%s changed=%s\n", ...
    amendment.sequence,string(metadata.batchId), ...
    strjoin(string({changedFunctions.name}),", "));
end

function changed = allowedChanges(fromSnapshot,toSnapshot)
assert(isequal(fieldnames(fromSnapshot),fieldnames(toSnapshot)), ...
    "ARES:AmendmentChangesFrozenSource", "Source snapshot structure changed.");
assert(isequal(rmfield(fromSnapshot,"collectionCode"), ...
    rmfield(toSnapshot,"collectionCode")), ...
    "ARES:AmendmentChangesFrozenSource", ...
    "Model, initializer, configuration, helpers, MATLAB or toolbox versions changed; start a new batch.");
before = fromSnapshot.collectionCode;
after = toSnapshot.collectionCode;
assert(isequal(size(before),size(after)),"ARES:AmendmentChangesFrozenSource", ...
    "Collection-code dependency list changed.");
allowed = ["captureLidarAxisBatch","buildLidarEpisode"];
changed = repmat(struct("name","","oldSha256","","newSha256",""),0,1);
for k=1:numel(before)
    assert(isequal(rmfield(before(k),"sha256"),rmfield(after(k),"sha256")), ...
        "ARES:AmendmentChangesFrozenSource", "Collection-code dependency identity changed.");
    oldHash = string(before(k).sha256);
    newHash = string(after(k).sha256);
    if oldHash==newHash
        continue;
    end
    name = string(before(k).name);
    assert(ismember(name,allowed),"ARES:AmendmentChangesFrozenSource", ...
        "Only the capture runner and episode builder may be amended; changed: %s",name);
    changed(end+1,1) = struct("name",name,"oldSha256",oldHash,"newSha256",newHash); %#ok<AGROW>
end
end

function hash = fileHash(path)
fid = fopen(path,"rb");
assert(fid~=-1,"ARES:SourceReadFailed","Cannot read source: %s",path);
cleanup = onCleanup(@()fclose(fid));
digest = java.security.MessageDigest.getInstance("SHA-256");
while ~feof(fid)
    bytes = fread(fid,1048576,"*uint8");
    digest.update(bytes);
end
hash = lower(string(reshape(dec2hex(typecast(digest.digest(),"uint8"),2).',1,[])));
end

function writeJsonExclusive(path,value)
assert(~isfile(path),"ARES:AmendmentAlreadyExists", ...
    "Amendment JSON already exists and will not be overwritten: %s",path);
fid = fopen(path,"w","n","UTF-8");
assert(fid~=-1,"ARES:WriteFailed","Cannot write amendment: %s",path);
cleanup = onCleanup(@()fclose(fid));
fprintf(fid,"%s",jsonencode(value,PrettyPrint=true));
end
