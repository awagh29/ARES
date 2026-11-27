function options = defaultOptions()
%DEFAULTOPTIONS Single-UAV map-temporal CNN experiment inputs (metres, NED).
root=string(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))));
common=ares.navigation.defaultOptions();
names=["StartState","GoalState","StartEulerZYXRad","StartBodyRatesRadps", ...
    "StopTimeS","ShowSimulation","LidarMode","NumberOfOutages","OutageDurationS", ...
    "InitialHealthyTimeS","FinalHealthyTimeS","MinimumRecoveryTimeS", ...
    "OutageWindowEndS","OutageIntervalsS","Seed","VehicleRadiusM", ...
    "MinimumSurfaceClearanceM","GoalPositionToleranceM","GoalVelocityToleranceMps"];
for name=names, options.(name)=common.(name); end
options.WorldBoundsNedM=[-5 40;-5 40;-10 -1]; % altitude 1--10 m
% Applied only to the experiment copy, after the original position PID.
options.EnableReferenceGovernor=true;
options.EnableLocalPlanner=true;
options.PlannerHorizonS=2.0;
options.PlannerRolloutStepS=0.2;
options.PlannerVoxelM=0.2;
options.PlannerStallTimeS=2.5;
options.PlannerProgressM=0.35;
options.PlannerRecoveryTimeS=6;
options.CruiseSpeedMps=1.5;
options.BrakingDecelerationMps2=0.8;
options.GovernorLatencyS=0.35; % scan/ZOH and tracking allowance, not wall inference
options.GovernorMarginM=0.35; % range/geometry and tracking allowance
options.GovernorPlanningBufferM=0.6; % steer before the braking envelope is tight
options.GoalApproachRadiusM=4;
options.GoalApproachGain=0.7;
options.GovernorAccelerationMps2=0.8;
options.ModelPath=""; % auto-select newest downloaded run/best.pt; or paste .pt/.mat
if ispc
    options.PythonExecutable=fullfile(root,'.venv','Scripts','python.exe');
else
    options.PythonExecutable=fullfile(root,'.venv','bin','python');
end
end
