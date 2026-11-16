function prepared = prepareModel(projectRoot,options)
%PREPAREMODEL Plan, wire, compile and save a standalone experiment copy.
arguments
    projectRoot (1,1) string
    options (1,1) struct = struct()
end
options=ares.navigation.validateOptions(options);
baseline=fullfile(projectRoot,'models','ares_single_uav_top.slx');
baselineHash=ares.inference.fileSha256(baseline);
model="ares_single_uav_map_correction";
if bdIsLoaded(model)
    assert(string(get_param(model,'SimulationStatus'))=="stopped" && string(get_param(model,'Dirty'))=="off", ...
        'ARES:MapModelHasEdits','Stop and save the open map-correction model before preparing another mission.');
    close_system(model,0);
end
assert(~bdIsLoaded('ares_single_uav_top') || ...
    string(get_param('ares_single_uav_top','SimulationStatus'))=="stopped", ...
    'ARES:BaselineRunning','Stop the baseline simulation before initializing another scene.');
folder=fullfile(projectRoot,'results','map_correction',string(java.util.UUID.randomUUID()));
mkdir(folder);
planFile=fullfile(folder,'mission.mat');
[map,cfg,schedule,intervals]=ares.navigation.initializeSimulation(projectRoot,planFile,options);
fprintf('Planning 3-D route in the existing obstacle map...\n');
plan=ares.navigation.planPath(map,options);
% Reject an initially inadmissible velocity/position before starting the plant.
[~,initial]=ares.navigation.correctMotion(options.StartState(1:3),options.StartState(4:6), ...
    [0 0 0],[0 0 0],map,options);
assert(initial.Feasible,'ARES:MapInitialStateUnsafe','Initial state fails the map barrier/actuation constraints.');
save(planFile,'options','map','cfg','plan','intervals');
modelPath=fullfile(folder,model+'.slx'); copyfile(baseline,modelPath);
load_system(modelPath);
try
    wiring=ares.navigation.configureModel(model,planFile,options);
    rootQuoted="'"+replace(projectRoot,"'","''")+"'";
    planQuoted="'"+replace(planFile,"'","''")+"'";
    set_param(model,'InitFcn',"ares.navigation.initializeSimulation("+rootQuoted+","+planQuoted+");", ...
        'StopFcn','');
    set_param(model,'SimulationCommand','update');
    save_system(model,modelPath);
catch exception
    if bdIsLoaded(model), close_system(model,0); end
    rethrow(exception);
end
assert(ares.inference.fileSha256(baseline)==baselineHash,'ARES:BaselineChanged','Baseline changed.');
prepared=struct('SchemaVersion',"ares-map-correction-preparation-v1", ...
    'ProjectRoot',projectRoot,'ModelName',model,'ModelPath',modelPath,'PlanFile',planFile, ...
    'BaselineSha256',baselineHash,'PreparedModelSha256',ares.inference.fileSha256(modelPath), ...
    'Options',options,'Plan',plan,'OutageIntervalsS',intervals, ...
    'ScheduleTimeS',double(schedule.Time),'ScheduleEnabled',double(schedule.Data),'Wiring',wiring);
files=dir(fullfile(projectRoot,'src','+ares','+navigation','*.m'));
paths=["scripts/initializeSingleUav.m";"src/+ares/+config/defaultSingleUav.m"; ...
    "src/+ares/+inference/sectorFeatures.m";"src/+ares/+data/evaluateAxisRun.m"; ...
    "src/+ares/+data/timeseriesToSignal.m";"src/+ares/+navigation/"+string({files.name})'];
prepared.RuntimeSources=struct('Path',{},'Sha256',{});
for k=1:numel(paths)
    prepared.RuntimeSources(k)=struct('Path',paths(k),'Sha256',ares.inference.fileSha256(fullfile(projectRoot,paths(k))));
end
prepared.MissionSha256=ares.inference.fileSha256(planFile);
writeJson(fullfile(folder,'prepared.json'),prepared);
writeJson(fullfile(projectRoot,'results','map_correction','latest_prepared.json'),prepared);
assignin('base','mapCorrectionPrepared',prepared);
if options.ShowSimulation, open_system(model); end
fprintf('Prepared %s: %.2f m route, %d waypoints, %d planned outage(s).\n', ...
    model,plan.LengthM,size(plan.WaypointsNedM,1),size(intervals,1));
fprintf('Saved model: %s\n',modelPath);
end
function writeJson(path,value)
fid=fopen(path,'w'); assert(fid>=0,'ARES:MapWriteFailed','Cannot write %s',path);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
