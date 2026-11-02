function runLidarAxisWorker(projectRoot,batchPath,workerId,runIndices)
%RUNLIDARAXISWORKER Run one disjoint shard with isolated Simulink caches.
arguments
    projectRoot (1,1) string
    batchPath (1,1) string
    workerId (1,1) string
    runIndices (1,:) double {mustBeInteger,mustBePositive}
end
assert(~isempty(regexp(workerId,"^worker[0-9]+$","once")), ...
    "ARES:InvalidWorkerId","Use a worker ID such as worker01.");
cacheRoot=fullfile(batchPath,"cache",workerId);
Simulink.fileGenControl("set","CacheFolder",fullfile(cacheRoot,"simulink"), ...
    "CodeGenFolder",fullfile(cacheRoot,"codegen"),"createDir",true);
[~,index]=captureLidarAxisBatch(projectRoot,ResumeBatchPath=batchPath, ...
    WorkerId=workerId,RunIndices=runIndices);
assert(all(index.Status(runIndices)=="complete"),"ARES:WorkerIncomplete", ...
    "Worker stopped before every assigned run completed; inspect its index and attempt log.");
fprintf("WORKER_COMPLETE %s runs=%d\n",workerId,numel(runIndices));
end
