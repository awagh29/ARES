function options = validateOptions(input)
%VALIDATEOPTIONS Validate mission inputs without exposing unused CBF settings.
options=ares.mapTemporal.defaultOptions();
for name=string(fieldnames(input))'
    assert(isfield(options,name),'ARES:UnknownTemporalOption','Unknown map-temporal option: %s',name);
    options.(name)=input.(name);
end
common=ares.mapTemporal.navigationOptions(options);
validated=ares.navigation.validateOptions(common);
for name=string(fieldnames(options))'
    if isfield(validated,name), options.(name)=validated.(name); end
end
trained=[-5 40;-5 40;-10 -1];
assert(all(options.WorldBoundsNedM(:,1)>=trained(:,1)) && ...
    all(options.WorldBoundsNedM(:,2)<=trained(:,2)), ...
    'ARES:TemporalTrainingBounds','Use North/East -5..40 m and altitude 1..10 m, the trained domain.');
for name=["StartState","GoalState"]
    p=options.(name)(1:3)';
    assert(all(p>=options.WorldBoundsNedM(:,1) & p<=options.WorldBoundsNedM(:,2)), ...
        'ARES:TemporalStateBounds','%s position is outside WorldBoundsNedM.',name);
end
options.ModelPath=string(options.ModelPath); options.PythonExecutable=string(options.PythonExecutable);
assert(islogical(options.EnableReferenceGovernor) && isscalar(options.EnableReferenceGovernor), ...
    'ARES:TemporalGovernorOption','EnableReferenceGovernor must be a logical scalar.');
assert(islogical(options.EnableLocalPlanner) && isscalar(options.EnableLocalPlanner), ...
    'ARES:TemporalPlannerOption','EnableLocalPlanner must be a logical scalar.');
for name=["PlannerHorizonS","PlannerRolloutStepS","PlannerVoxelM","PlannerStallTimeS", ...
        "PlannerProgressM","PlannerRecoveryTimeS"]
    value=options.(name);
    assert(isnumeric(value) && isreal(value) && isscalar(value) && isfinite(value) && value>0, ...
        'ARES:TemporalPlannerOption','%s must be a positive finite scalar.',name);
end
assert(options.PlannerRolloutStepS<=.25 && options.PlannerHorizonS>=1 && options.PlannerHorizonS<=4 && ...
    options.PlannerVoxelM<=.3 && options.PlannerStallTimeS>=1 && options.PlannerRecoveryTimeS>=2, ...
    'ARES:TemporalPlannerOption','Planner settings exceed the supported sampling bounds.');
for name=["CruiseSpeedMps","BrakingDecelerationMps2","GovernorLatencyS", ...
        "GovernorMarginM","GovernorPlanningBufferM","GoalApproachRadiusM","GoalApproachGain","GovernorAccelerationMps2"]
    value=options.(name);
    assert(isnumeric(value) && isreal(value) && isscalar(value) && isfinite(value) && value>0, ...
        'ARES:TemporalGovernorOption','%s must be a positive finite scalar.',name);
end
assert(options.GoalApproachRadiusM<=5 && options.GoalApproachRadiusM>options.GoalPositionToleranceM, ...
    'ARES:TemporalGovernorOption','GoalApproachRadiusM must exceed the goal tolerance and be <=5 m.');
assert(isscalar(options.ModelPath) && isscalar(options.PythonExecutable), ...
    'ARES:TemporalModelPath','ModelPath and PythonExecutable must be scalar paths.');
end
