function report = validateSaitsBarrierMatlab(projectRoot,deploymentPath,pythonExecutable)
%VALIDATESAITSBARRIERMATLAB Numerical bridge checks only; no plant simulation.
arguments
    projectRoot (1,1) string
    deploymentPath (1,1) string
    pythonExecutable (1,1) string
end
module=ares.inference.loadSaitsBarrierRuntime(projectRoot,pythonExecutable);
goal=[30 35 -7];
runtime=module.Controller(char(deploymentPath),goal);
report=jsondecode(char(runtime.reference_check()));
assert(report.passed,"ARES:BarrierReferenceParity","Selected deployment failed its exported numerical fixtures.");
controller=ares.inference.SaitsBarrierController("ProjectRoot",char(projectRoot), ...
    "DeploymentPath",char(deploymentPath),"PythonExecutable",char(pythonExecutable),"GoalNed",goal);
cleanup=onCleanup(@()release(controller)); %#ok<NASGU>
cloud=NaN(10740,3,"single");
p=[0 0 -7]; v=zeros(1,3); rpy=zeros(1,3); w=zeros(1,3);
[~,command,packet,failed]=controller(cloud,p,v,rpy,w,true,0);
expected=reshape(double(runtime.step(p,v,rpy,w,[7*ones(1,7) zeros(1,7)],true,0)),113,1);
indices=[1:110 112:113]; % Runtime durations are measurements, not predictions.
report.matlabPacketMaximumError=max(abs(packet(indices)-expected(indices)));
assert(report.matlabPacketMaximumError<1e-5, ...
    "ARES:BarrierBridgeParity","MATLAB/Python signal layout differs.");
assert(isequal(command,single(packet(1:4))),"ARES:BarrierCommandParity","Executed commands differ from runtime.");
assert(failed==logical(packet(110)),"ARES:BarrierStopParity","Failure stop signal is not the runtime failure latch.");
[~,~,duplicate]=controller(cloud,p,v,rpy,w,true,0);
assert(isequal(packet,duplicate),"ARES:DuplicateBarrierTick","Repeated output evaluation changed history.");
% Poison the outage cloud. Neither MATLAB nor Python may inspect its values.
[~,~,outage]=controller(ones(10740,3,"single"),p,v,rpy,w,false,.1);
expected=reshape(double(runtime.step(p,v,rpy,w,NaN(1,14),false,.1)),113,1);
indices=[1:110 112:113];
report.outageBridgeMaximumError=max(abs(outage(indices)-expected(indices)));
assert(report.outageBridgeMaximumError<1e-5,"ARES:BarrierOutageParity","Outage branch differs from trained runtime.");
assert(all(outage(64:77)==0),"ARES:BarrierOracleLeak","Unavailable LiDAR was marked measured.");
% Explicit continuation must not activate Stop Simulation, even though the
% runtime retains its failure latch and false feasibility for this near scan.
continuing=ares.inference.SaitsBarrierController("ProjectRoot",char(projectRoot), ...
    "DeploymentPath",char(deploymentPath),"PythonExecutable",char(pythonExecutable), ...
    "GoalNed",goal,"ContinueAfterFailure",true);
continuingCleanup=onCleanup(@()release(continuing)); %#ok<NASGU>
nearCloud=zeros(10740,3,"single"); nearCloud(:,1)=single(.1);
[~,~,diagnostic,stopRequested]=continuing(nearCloud,p,v,rpy,w,true,0);
assert(~stopRequested && diagnostic(110)==1 && diagnostic(106)==0, ...
    "ARES:DiagnosticContinuationStopped","Diagnostic mode must retain failure without requesting Stop Simulation.");
assert(max(abs(diagnostic(98:99)))<=.100001 && diagnostic(98)<-.09, ...
    "ARES:DiagnosticFallbackIncorrect","Expected slew-bounded perceived-LiDAR nominal fallback.");
[~,~,nextDiagnostic,nextStop]=continuing(nearCloud,p,v,rpy,w,true,.1);
assert(~nextStop && nextDiagnostic(110)==1, ...
    "ARES:DiagnosticContinuationLatchedStop","Diagnostic run must accept a later scan after certificate failure.");
report.diagnosticContinuationChecked=true;
runtime.reset();
durations=zeros(5,1);
for k=1:5
    timer=tic;
    runtime.step(p,v,rpy,w,[7*ones(1,7) zeros(1,7)],true,(k-1)*.1);
    durations(k)=toc(timer);
end
report.matlabRoundTripMedianMilliseconds=1000*median(durations);
report.matlabRoundTripMaximumMilliseconds=1000*max(durations);
report.performanceIsHardRealtimeGuarantee=false;
report.matlabRelease=string(version("-release"));
report.pythonExecutable=pythonExecutable;
report.deploymentManifestSha256=ares.inference.fileSha256(fullfile(deploymentPath,"deployment.json"));
report.schemaVersion="ares-saits-barrier-matlab-validation-v1";
report.plantSimulationRun=false;
report.passed=true;
fprintf("Augmented SAITS/BarrierNet bridge validated; median MATLAB round trip %.2f ms. No plant simulation run.\n", ...
    report.matlabRoundTripMedianMilliseconds);
end
