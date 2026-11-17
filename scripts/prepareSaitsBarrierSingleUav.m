%PREPARESAITSBARRIERSINGLEUAV Validate and compile the trained model; do not run.
project=currentProject;
projectRoot=string(project.RootFolder);
if exist("SaitsDeploymentPathOverride","var")
    deploymentPath=string(SaitsDeploymentPathOverride);
else
    deploymentPath=fullfile(projectRoot,"results","augmented_saits_barrier", ...
        "ares_augmented_saits_barrier_local_seed42_v1","deployment");
end
if exist("SaitsPythonExecutableOverride","var")
    pythonExecutable=string(SaitsPythonExecutableOverride);
else
    pythonExecutable=fullfile(projectRoot,".venv","Scripts","python.exe");
end
if exist("SaitsShowSimulationOverride","var")
    showSimulation=logical(SaitsShowSimulationOverride);
else
    showSimulation=usejava("desktop");
end
if exist("SaitsContinueAfterBarrierFailureOverride","var")
    continueAfterFailure=logical(SaitsContinueAfterBarrierFailureOverride);
else
    continueAfterFailure=true; % Requested diagnostic run: keep pursuing the goal.
end
saitsBarrierPrepared=ares.inference.prepareSaitsBarrierModel( ...
    projectRoot,deploymentPath,pythonExecutable,showSimulation,continueAfterFailure);
