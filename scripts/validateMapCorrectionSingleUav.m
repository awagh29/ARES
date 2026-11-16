function report = validateMapCorrectionSingleUav(projectRoot)
%VALIDATEMAPCORRECTIONSINGLEUAV Real-plant healthy/outage/cold-blackout checks.
arguments
    projectRoot (1,1) string = string(fileparts(fileparts(mfilename('fullpath'))))
end
addpath(fullfile(projectRoot,'src'));
unit=runtests(fullfile(projectRoot,'tests','data','TestMapCorrection.m'));
assertSuccess(unit);
o=ares.navigation.defaultOptions(); o.ShowSimulation=false;
modes=["healthy","intermittent","blackout"];
report=struct('SchemaVersion',"ares-map-correction-validation-v1", ...
    'UnitTestsPassed',nnz([unit.Passed]),'Cases',struct());
for mode=modes
    o.LidarMode=mode;
    prepared=ares.navigation.prepareModel(projectRoot,o);
    result=ares.navigation.runMission(prepared);
    assert(result.Metrics.missionSuccess,'ARES:MapValidationMission','%s mission failed.',mode);
    assert(result.Proof.AvailabilityMatchesSchedule && result.Proof.EveryOutageUsesMapCorrection && ...
        result.Proof.EveryHealthyTickUsesMeasuredLidar && result.Proof.AllFiltersFeasible, ...
        'ARES:MapValidationProof','%s instrumentation failed.',mode);
    if mode=="intermittent"
        assert(result.ObservedOutageCount==o.NumberOfOutages,'ARES:MapValidationCount','Not every default outage was encountered.');
    elseif mode=="blackout"
        assert(result.OutageControlTicks==result.ControlTicks,'ARES:MapValidationBlackout','Blackout used a healthy tick.');
    end
    ares.navigation.plotRun(result);
    report.Cases.(mode)=result;
end
% Different altitude endpoints and a moving/yawed initial state exercise the
% actual 3-D plant initialization, not only the geometry/planner unit test.
o.StartState=[2 -3 -6 .2 0 0]; o.GoalState=[26 29 -17 0 0 0];
o.StartEulerZYXRad=[.3 0 0]; o.LidarMode="blackout";
prepared=ares.navigation.prepareModel(projectRoot,o);
result=ares.navigation.runMission(prepared);
assert(result.Metrics.missionSuccess && result.Proof.InitialVelocityMatchesRequest, ...
    'ARES:MapValidationCustomState','The custom 3-D state mission failed.');
ares.navigation.plotRun(result); report.Cases.custom3d=result;
report.Passed=true;
report.Path=fullfile(projectRoot,'results','map_correction','validation.json');
fid=fopen(report.Path,'w'); assert(fid>=0,'ARES:MapWriteFailed','Cannot write validation report.');
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
fprintf('PASS: %d unit tests and four full Simulink missions.\n',report.UnitTestsPassed);
end
