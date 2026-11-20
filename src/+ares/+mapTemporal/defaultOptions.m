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
options.ModelPath=""; % auto-select newest downloaded run/best.pt; or paste .pt/.mat
if ispc
    options.PythonExecutable=fullfile(root,'.venv','Scripts','python.exe');
else
    options.PythonExecutable=fullfile(root,'.venv','bin','python');
end
end
