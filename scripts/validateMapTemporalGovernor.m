function report = validateMapTemporalGovernor(projectRoot,modelPath,modes)
%VALIDATEMAPTEMPORALGOVERNOR Paired frozen-model real-plant comparisons.
% Both arms use identical mission inputs, outage schedules and native weights.
% Report failures as outcomes; do not hide an unsuccessful controller flight.
arguments
    projectRoot (1,1) string = string(fileparts(fileparts(mfilename('fullpath'))))
    modelPath (1,1) string = ""
    modes (1,:) string = ["healthy","intermittent","blackout"]
end
addpath(fullfile(projectRoot,'src'));
assert(all(ismember(modes,["healthy","intermittent","blackout"])), ...
    'ARES:TemporalValidationMode','Choose healthy, intermittent or blackout.');
o=ares.mapTemporal.defaultOptions(); o.ShowSimulation=false; o.ModelPath=modelPath;
o.EnableLocalPlanner=false; % this comparison isolates the original speed governor
artifact=ares.mapTemporal.resolveArtifact(projectRoot,o);
o.ModelPath=artifact.Path; % resolve/export once; all flights use this exact file
weightHash=ares.inference.fileSha256(artifact.Path);
baselineHash=ares.inference.fileSha256(fullfile(projectRoot,'models','ares_single_uav_top.slx'));
folder=fullfile(projectRoot,'results','map_temporal_governor',string(java.util.UUID.randomUUID()));
mkdir(folder);
report=struct('SchemaVersion',"ares-map-temporal-governor-validation-v1", ...
    'Artifact',artifact,'NativeWeightSha256',weightHash,'BaselineModelSha256',baselineHash, ...
    'Directory',folder,'Cases',struct());
unit=runtests([fullfile(projectRoot,'tests','data','TestMapTemporalGovernor.m'), ...
    fullfile(projectRoot,'tests','data','TestMapTemporal.m')]); assertSuccess(unit);
report.UnitTestsPassed=nnz([unit.Passed]);
for mode=modes
    o.LidarMode=mode;
    for enabled=[false true]
        o.EnableReferenceGovernor=enabled;
        name="baseline"; if enabled, name="governed"; end
        fprintf('\nCOMPARISON %s / %s\n',mode,name);
        prepared=ares.mapTemporal.prepareModel(projectRoot,o);
        result=ares.mapTemporal.runMission(prepared);
        report.Cases.(mode).(name)=result;
        assert(ares.inference.fileSha256(artifact.Path)==weightHash, ...
            'ARES:TemporalWeightsChanged','Frozen native weights changed.');
        ares.mapTemporal.writeJson(fullfile(folder,'comparison.json'),report);
    end
    b=report.Cases.(mode).baseline; g=report.Cases.(mode).governed;
    bp=load(b.PlanFile,'intervals'); gp=load(g.PlanFile,'intervals');
    assert(isequal(bp.intervals,gp.intervals),'ARES:TemporalComparisonSchedule','Schedules differ.');
    report.Cases.(mode).OutageSchedulesIdentical=true;
    report.Cases.(mode).Change=struct( ...
        'MinimumClearanceM',g.Metrics.minimumObstacleClearanceM-b.Metrics.minimumObstacleClearanceM, ...
        'FinalPositionErrorM',g.Metrics.finalWaypointErrorM-b.Metrics.finalWaypointErrorM, ...
        'FinalVelocityErrorMps',g.Metrics.finalVelocityErrorMps-b.Metrics.finalVelocityErrorMps);
    comparisonPlot(b,g,fullfile(folder,mode+'.png'));
end
assert(ares.inference.fileSha256(fullfile(projectRoot,'models','ares_single_uav_top.slx'))==baselineHash, ...
    'ARES:BaselineChanged','Original Simulink model changed.');
report.FrozenWeightsVerified=true; report.BaselineModelUnchanged=true;
ares.mapTemporal.writeJson(fullfile(folder,'comparison.json'),report);
fprintf('Comparison saved: %s\n',folder);
end

function comparisonPlot(b,g,path)
first=readtable(fullfile(b.Directory,'scans.csv')); second=readtable(fullfile(g.Directory,'scans.csv'));
f=figure('Visible','off','Color','w','Position',[100 100 1100 750]);
clean=onCleanup(@()close(f)); %#ok<NASGU>
tiledlayout(f,2,2,'TileSpacing','compact');
ax=nexttile; plot(ax,first.position_n,first.position_e,'--',second.position_n,second.position_e,'LineWidth',1.3);
hold(ax,'on'); mission=load(g.PlanFile,'map');
for k=1:size(mission.map.BoxMin,1)
    lo=mission.map.BoxMin(k,:); hi=mission.map.BoxMax(k,:);
    rectangle(ax,'Position',[lo(1:2),hi(1:2)-lo(1:2)],'EdgeColor',[.4 .4 .4]);
end
axis(ax,'equal'); grid(ax,'on'); xlabel(ax,'North (m)'); ylabel(ax,'East (m)'); title(ax,'Actual flight');
legend(ax,'Original','Governor','Location','best');
ax=nexttile; plot(ax,first.time_s,first.goal_position_error_m,'--',second.time_s,second.goal_position_error_m,'LineWidth',1.3);
grid(ax,'on'); xlabel(ax,'Time (s)'); ylabel(ax,'Goal position error (m)');
ax=nexttile; plot(ax,first.time_s,first.goal_velocity_error_mps,'--',second.time_s,second.goal_velocity_error_mps,'LineWidth',1.3);
grid(ax,'on'); xlabel(ax,'Time (s)'); ylabel(ax,'Goal velocity error (m/s)');
ax=nexttile; stairs(ax,second.time_s,second.lidar_available,'LineWidth',1.3);
ylim(ax,[-.1 1.1]); grid(ax,'on'); xlabel(ax,'Time (s)'); ylabel(ax,'Measured LiDAR available');
sgtitle(f,sprintf('%s: original versus braking/terminal governor (frozen reconstruction)',g.LidarMode));
for ax=findall(f,'Type','axes')'
    set(ax,'Color','w','XColor','k','YColor','k');
    set([ax.Title,ax.XLabel,ax.YLabel],'Color','k');
end
set(findall(f,'Type','legend'),'Color','w','TextColor','k');
exportgraphics(f,path,'Resolution',150);
end
