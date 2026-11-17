function module = loadSaitsBarrierRuntime(projectRoot, pythonExecutable)
%LOADSAITSBARRIERRUNTIME Isolate the selected environment from MATLAB DLLs.
arguments
    projectRoot (1,1) string
    pythonExecutable (1,1) string
end
assert(isfile(pythonExecutable),"ARES:MissingBarrierPython","Missing Python: %s",pythonExecutable);
environment = pyenv;
if string(environment.Status)=="Loaded"
    assert(strcmpi(string(environment.Executable),pythonExecutable) && ...
        string(environment.ExecutionMode)=="OutOfProcess", ...
        "ARES:BarrierPythonAlreadyLoaded", ...
        "MATLAB has another Python environment loaded. Restart MATLAB, then run prepareSaitsBarrierSingleUav.m.");
else
    % Resolve/launch the virtual-environment redirector before MATLAB queries
    % its base DLL. Also report a clear interpreter error before importing.
    [status,versionText]=system('"'+pythonExecutable+'" --version');
    assert(status==0,"ARES:BarrierPythonLaunchFailed","Cannot launch %s: %s",pythonExecutable,string(versionText));
    environment=pyenv(Version=char(pythonExecutable),ExecutionMode="OutOfProcess");
    assert(strcmpi(string(environment.Executable),pythonExecutable), ...
        "ARES:WrongBarrierPython","MATLAB selected another Python interpreter.");
end
scripts = char(fullfile(projectRoot,"scripts"));
pythonPath=py.sys.path;
if int64(pythonPath.count(scripts))==0
    pythonPath.insert(int32(0),scripts);
end
module = py.importlib.import_module("saits_barrier_simulink");
% A previous capture can leave the old module cached in this MATLAB session.
% Refresh changed bridge code without discarding the user's Python host.
expectedSource=ares.inference.fileSha256(fullfile(projectRoot,"scripts","saits_barrier_simulink.py"));
if ~logical(py.hasattr(module,"RUNTIME_SOURCE_SHA256")) || string(module.RUNTIME_SOURCE_SHA256)~=expectedSource
    py.importlib.invalidate_caches();
    module=py.importlib.reload(module);
end
end
