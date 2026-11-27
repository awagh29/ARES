function result = runMission(prepared)
%RUNMISSION Run the real plant and report measured/learned behavior and errors.
data=load(prepared.PlanFile,'options','map','cfg'); o=data.options;
baseline=fullfile(prepared.ProjectRoot,'models','ares_single_uav_top.slx');
assert(ares.inference.fileSha256(baseline)==prepared.BaselineSha256,'ARES:BaselineChanged','Baseline changed since preparation.');
assert(ares.inference.fileSha256(prepared.ModelPath)==prepared.PreparedModelSha256, ...
    'ARES:TemporalModelChanged','Prepared model changed. Prepare the mission again.');
assert(ares.inference.fileSha256(prepared.PlanFile)==prepared.MissionSha256, ...
    'ARES:TemporalMissionChanged','Mission inputs changed. Prepare the mission again.');
assert(ares.inference.fileSha256(prepared.Artifact.Path)==prepared.Artifact.Sha256, ...
    'ARES:TemporalWeightsChanged','Native weights changed. Prepare the mission again.');
for source=prepared.RuntimeSources
    assert(ares.inference.fileSha256(fullfile(prepared.ProjectRoot,source.Path))==source.Sha256, ...
        'ARES:TemporalRuntimeChanged','Source changed since preparation: %s. Prepare again.',source.Path);
end
load_system(prepared.ModelPath);
timer=tic; simOut=sim(prepared.ModelName); wall=toc(timer);
% Preserve the expensive plant run before any optional analysis/plotting.
save(fullfile(fileparts(prepared.PlanFile),'simulation.mat'),'simOut','prepared','wall','-v7.3');
logs=simOut.logsout;
packet=signal(logs,'map_temporal_runtime_packet',24);
time=double(packet.time_s(:)); values=double(packet.values);
telemetry.signals.position_ned_m=signal(logs,'position_ned_m',3);
telemetry.signals.velocity_ned_mps=signal(logs,'velocity_ned_mps',3);
centers=(data.map.BoxMin+data.map.BoxMax)/2;
metrics=ares.data.evaluateAxisRun(telemetry,data.cfg,centers,3,15);
expected=interp1(prepared.ScheduleTimeS,prepared.ScheduleEnabled,time,'previous')>=.5;
inputs=signal(logs,'map_temporal_input_features',10752);
support=signal(logs,'map_temporal_input_observed_mask',10752);
hits=signal(logs,'map_temporal_input_valid_return_mask',10740);
dense=signal(logs,'map_temporal_predicted_ranges_m',10740);
nominal=signal(logs,'map_temporal_nominal_ranges_scoring',10740);
assert(isequal(inputs.time_s(:),time) && isequal(support.time_s(:),time) && ...
    isequal(hits.time_s(:),time) && isequal(dense.time_s(:),time) && isequal(nominal.time_s(:),time), ...
    'ARES:TemporalLogAlignment','Expected matching 10 Hz inference/scoring logs.');
proof=struct('AvailabilityMatchesSchedule',isequal(logical(values(:,13)),expected), ...
    'EveryOutageUsesLearnedReconstruction',all(values(~expected,14)==1), ...
    'EveryHealthyTickUsesMeasuredLidar',all(values(expected,14)==0), ...
    'OutageInputRangesZero',all(inputs.values(~expected,1:10740)==0,'all'), ...
    'OutageInputSupportFalse',~any(support.values(~expected,1:10740),'all'), ...
    'OutageInputHitsFalse',~any(hits.values(~expected,:),'all'), ...
    'FreshHistoryAtStart',values(1,15)==1,'FiniteSelectedRanges',all(isfinite(dense.values),'all'));
velocity=double(telemetry.signals.velocity_ned_mps.values);
proof.InitialVelocityMatchesRequest=norm(velocity(1,:)-o.StartState(4:6))<1e-4;
goalAtTermination=logical(values(end,20));
metrics.finalVelocityErrorMps=norm(velocity(end,:)-o.GoalState(4:6));
metrics.goalReachedAtTermination=metrics.goalReachedAtTermination && goalAtTermination && ...
    metrics.finalVelocityErrorMps<=o.GoalVelocityToleranceMps;
metrics.missionSuccess=metrics.missionSuccess && metrics.goalReachedAtTermination && ...
    all(structfun(@(v)logical(v),proof));
if ~metrics.goalReachedAtTermination && metrics.reachedTimeLimit
    metrics.terminationReason="time-limit-before-terminal-state";
elseif metrics.goalReachedAtTermination
    metrics.terminationReason="goal-state-reached";
end
metrics.timedOut=metrics.reachedTimeLimit && ~metrics.goalReachedAtTermination;
errorM=double(dense.values(~expected,:))-double(nominal.values(~expected,:));
outageRMSE=NaN; outageMAE=NaN;
if ~isempty(errorM), outageRMSE=sqrt(mean(errorM.^2,'all')); outageMAE=mean(abs(errorM),'all'); end
result=struct('SchemaVersion',"ares-map-temporal-run-v1", ...
    'ModelPath',prepared.ModelPath,'PlanFile',prepared.PlanFile,'Artifact',prepared.Artifact, ...
    'LidarMode',o.LidarMode,'PlannedOutageCount',size(prepared.OutageIntervalsS,1), ...
    'ObservedOutageCount',double(~expected(1))+nnz(diff(expected)==-1), ...
    'OutageScans',nnz(~expected),'Scans',numel(time),'FinalTimeS',time(end),'WallTimeS',wall, ...
    'Metrics',metrics,'Proof',proof,'OutageRangeRMSEM',outageRMSE,'OutageRangeMAEM',outageMAE, ...
    'MaximumInferenceS',max(values(:,17)),'SafetyCertified',false);
result.Directory=string(fileparts(prepared.PlanFile));
if o.EnableReferenceGovernor
    governor=signal(logs,'map_temporal_governor_packet',12);
    assert(isequal(governor.time_s(:),time),'ARES:TemporalGovernorClock','Expected 10 Hz governor logs.');
    g=double(governor.values);
    result.Governor=struct('Enabled',true,'ApproachTicks',nnz(g(:,11)), ...
        'BrakingTicks',nnz(g(:,10)),'MaximumRequestedSpeedMps',max(vecnorm(g(:,1:3),2,2)), ...
        'MaximumCommandedSpeedMps',max(vecnorm(g(:,4:6),2,2)));
    writetable(array2table([time,g],'VariableNames',["time_s",prepared.Wiring.GovernorPacketOrder]), ...
        fullfile(result.Directory,'governor.csv'));
else
    result.Governor=struct('Enabled',false);
end
if o.EnableReferenceGovernor && o.EnableLocalPlanner
    planner=signal(logs,'map_temporal_planner_packet',12); q=double(planner.values);
    assert(isequal(planner.time_s(:),time),'ARES:TemporalPlannerClock','Expected 10 Hz planner logs.');
    result.Planner=struct('Enabled',true,'RecoveryEntries',max(q(:,3)), ...
        'RecoveryTicks',nnz(q(:,1)),'NoFeasibleTicks',nnz(q(:,11)), ...
        'MaximumComputeS',max(q(:,12)),'MeanComputeS',mean(q(:,12)));
    writetable(array2table([time,q],'VariableNames',["time_s",prepared.Wiring.PlannerPacketOrder]), ...
        fullfile(result.Directory,'planner.csv'));
else
    result.Planner=struct('Enabled',false);
end
names=["position_n","position_e","position_d","velocity_n","velocity_e","velocity_d", ...
    "roll_rad","pitch_rad","yaw_rad","goal_n","goal_e","goal_d", ...
    "lidar_available","reconstruction_used","history_steps","blackout_age_s", ...
    "inference_s","goal_position_error_m","goal_velocity_error_mps","terminal_state_reached", ...
    "prior_min_range_m","residual_max_abs_m","scan_count","input_support_fraction"];
scans=array2table([time values],'VariableNames',["time_s",names]);
writetable(scans,fullfile(result.Directory,'scans.csv'));
save(fullfile(result.Directory,'telemetry.mat'),'telemetry','scans','dense','nominal','inputs','support','hits','prepared','result','-v7.3');
ares.mapTemporal.writeJson(fullfile(result.Directory,'summary.json'),result);
ares.mapTemporal.writeJson(fullfile(prepared.ProjectRoot,'results','map_temporal_simulation','latest_run.json'),result);
ares.mapTemporal.plotRun(telemetry,scans,data.map,o,result.Directory);
assert(ares.inference.fileSha256(baseline)==prepared.BaselineSha256,'ARES:BaselineChanged','Baseline changed during run.');
assignin('base','mapTemporalResult',result);
fprintf('Goal reached at rest: %d. Mission success: %d. Simulated time: %.2f s.\n', ...
    metrics.goalReachedAtTermination,metrics.missionSuccess,time(end));
fprintf('Minimum surface clearance: %.3f m. Outage range RMSE: %.4f m.\n',metrics.minimumObstacleClearanceM,outageRMSE);
fprintf('Observed/planned outages: %d/%d. Results: %s\n',result.ObservedOutageCount,result.PlannedOutageCount,result.Directory);
end
function s=signal(logs,name,width)
s=ares.data.timeseriesToSignal(logs.getElement(name).Values,width);
end
