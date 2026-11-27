function report=validateMapTemporalPlanner(projectRoot,modelPath,caseNames)
%VALIDATEMAPTEMPORALPLANNER Compare stalled governor missions with local planning.
% Reuse matching archived controls, or run them if no matching record exists.
arguments
    projectRoot (1,1) string = string(fileparts(fileparts(mfilename('fullpath'))))
    modelPath (1,1) string = ""
    caseNames (1,:) string = ["shortIntermittent","completeBlackout","screenshotIntermittent"]
end
addpath(fullfile(projectRoot,'src'));
assert(all(ismember(caseNames,["shortIntermittent","completeBlackout","screenshotIntermittent"])), ...
    'ARES:PlannerCase','Unknown planner comparison case.');
o=ares.mapTemporal.defaultOptions(); o.ShowSimulation=false; o.ModelPath=modelPath;
artifact=ares.mapTemporal.resolveArtifact(projectRoot,o); o.ModelPath=artifact.Path;
weights=ares.inference.fileSha256(artifact.Path);
original=ares.inference.fileSha256(fullfile(projectRoot,'models/ares_single_uav_top.slx'));
folder=fullfile(projectRoot,'results/map_temporal_planner',string(java.util.UUID.randomUUID())); mkdir(folder);
unit=runtests([fullfile(projectRoot,'tests/data/TestMapTemporal.m'), ...
    fullfile(projectRoot,'tests/data/TestMapTemporalGovernor.m'),fullfile(projectRoot,'tests/data/TestMapTemporalPlanner.m')]);
assertSuccess(unit);
report=struct('SchemaVersion',"ares-map-temporal-planner-validation-v1",'Directory',folder, ...
    'UnitTestsPassed',nnz([unit.Passed]),'NativeWeightSha256',weights,'BaselineModelSha256',original,'Cases',struct());
for name=caseNames
    o.StartState=[0 0 -7 0 0 0]; o.GoalState=[25 25 -7 0 0 0]; o.LidarMode="intermittent";
    if name=="shortIntermittent", o.StartState=[15 10 -7 0 0 0]; end
    if name=="completeBlackout", o.GoalState=[30 35 -7 0 0 0]; o.LidarMode="blackout"; end
    control=findControl(projectRoot,o,weights);
    if isempty(control)
        o.EnableLocalPlanner=false;
        control=ares.mapTemporal.runMission(ares.mapTemporal.prepareModel(projectRoot,o));
    end
    o.EnableLocalPlanner=true;
    fprintf('\nPLANNER COMPARISON: %s\n',name);
    result=ares.mapTemporal.runMission(ares.mapTemporal.prepareModel(projectRoot,o));
    before=load(control.PlanFile,'options','intervals'); after=load(result.PlanFile,'options','intervals');
    assert(isequal(before.intervals,after.intervals),'ARES:PlannerSchedule','Outage intervals differ.');
    assert(isequal(before.options.StartState,after.options.StartState) && ...
        isequal(before.options.GoalState,after.options.GoalState),'ARES:PlannerEndpoints','Endpoints differ.');
    assert(all(structfun(@(x)logical(x),result.Proof)),'ARES:PlannerIsolation','Input isolation check failed.');
    report.Cases.(name)=struct('governor',control,'planner',result,'OutageSchedulesIdentical',true);
    assert(ares.inference.fileSha256(artifact.Path)==weights,'ARES:TemporalWeightsChanged','Frozen weights changed.');
    ares.mapTemporal.writeJson(fullfile(folder,'comparison.json'),report);
    ares.mapTemporal.plotPlannerComparison(control,result,fullfile(folder,name+'.png'));
end
assert(ares.inference.fileSha256(fullfile(projectRoot,'models/ares_single_uav_top.slx'))==original,'ARES:BaselineChanged','Original model changed.');
report.FrozenWeightsVerified=true; report.BaselineModelUnchanged=true;
ares.mapTemporal.writeJson(fullfile(folder,'comparison.json'),report);
fprintf('Planner comparison: %s\n',folder);
end

function result=findControl(root,o,hash)
result=[]; files=dir(fullfile(root,'results/map_temporal_simulation/*/summary.json'));
[~,order]=sort([files.datenum],'descend');
original=ares.inference.fileSha256(fullfile(root,'models/ares_single_uav_top.slx'));
keys=["StartState","GoalState","LidarMode","StopTimeS","Seed","MinimumSurfaceClearanceM", ...
    "CruiseSpeedMps","BrakingDecelerationMps2","GovernorMarginM","GovernorPlanningBufferM", ...
    "GovernorLatencyS","GovernorAccelerationMps2","GoalApproachRadiusM","GoalApproachGain", ...
    "VehicleRadiusM","WorldBoundsNedM","GoalPositionToleranceM","GoalVelocityToleranceMps", ...
    "NumberOfOutages","OutageDurationS","InitialHealthyTimeS","FinalHealthyTimeS", ...
    "MinimumRecoveryTimeS","OutageWindowEndS","OutageIntervalsS"];
for index=order
    r=jsondecode(fileread(fullfile(files(index).folder,files(index).name)));
    if string(r.Artifact.Sha256)~=hash, continue; end
    metadata=jsondecode(fileread(fullfile(files(index).folder,'prepared.json')));
    if string(metadata.BaselineSha256)~=original, continue; end
    plan=load(r.PlanFile,'options'); p=plan.options;
    if ~isfield(p,'EnableReferenceGovernor') || ~p.EnableReferenceGovernor || ...
            (isfield(p,'EnableLocalPlanner') && p.EnableLocalPlanner), continue; end
    matched=true;
    for key=keys
        if ~isfield(p,key) || ~isequal(stringOrNumber(p.(key)),stringOrNumber(o.(key))), matched=false; break; end
    end
    if matched, result=r; return; end
end
end
function x=stringOrNumber(x)
if ischar(x) || isstring(x), x=string(x); end
end
