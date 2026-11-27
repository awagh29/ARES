function options = validateOptions(input)
%VALIDATEOPTIONS Merge partial inputs and reject ambiguous/unsafe missions.
options = ares.navigation.defaultOptions();
names = fieldnames(input);
for k=1:numel(names)
    assert(isfield(options,names{k}), 'ARES:UnknownMapOption', ...
        'Unknown map navigation option: %s',names{k});
    options.(names{k}) = input.(names{k});
end
vectors = ["StartState","GoalState","StartEulerZYXRad","StartBodyRatesRadps"];
for k=1:numel(vectors)
    name=vectors(k); n=3; if endsWith(name,"State"), n=6; end
    validateattributes(options.(name),{'numeric'},{'real','finite','size',[1 n]},mfilename,char(name));
    options.(name)=double(options.(name));
end
positive = ["StopTimeS","ControlSampleTimeS","GridResolutionM","VehicleRadiusM", ...
    "MinimumSurfaceClearanceM","PlanningExtraMarginM","CruiseSpeedMps", ...
    "BrakingDecelerationMps2","AccelerationLimitMps2","JerkLimitMps3", ...
    "VelocityGain","PositionGain","BarrierGain","GoalPositionToleranceM", ...
    "GoalVelocityToleranceMps","WaypointToleranceM","OutageWindowEndS"];
for name=positive
    validateattributes(options.(name),{'numeric'},{'real','finite','scalar','positive'},mfilename,char(name));
end
nonnegative=["GeometryUncertaintyM","TrackingMarginM","ClearanceWeight", ...
    "AccelerationUncertaintyMps2","InitialHealthyTimeS","FinalHealthyTimeS","MinimumRecoveryTimeS"];
for name=nonnegative
    validateattributes(options.(name),{'numeric'},{'real','finite','scalar','nonnegative'},mfilename,char(name));
end
validateattributes(options.NumberOfOutages,{'numeric'},{'real','finite','scalar','integer','nonnegative'});
validateattributes(options.Seed,{'numeric'},{'real','finite','scalar','integer','nonnegative','<=',2^32-1});
validateattributes(options.ShowSimulation,{'logical','numeric'},{'real','finite','scalar','binary'});
options.ShowSimulation=logical(options.ShowSimulation);
validateattributes(options.WorldBoundsNedM,{'numeric'},{'real','finite','size',[3 2]});
assert(all(diff(options.WorldBoundsNedM,1,2)>0),'ARES:InvalidMapBounds','World bounds must increase.');
assert(options.WorldBoundsNedM(3,2)<0,'ARES:MapBelowGround','Use negative Down coordinates above ground.');
assert(options.ControlSampleTimeS<=.1 && abs(.1/options.ControlSampleTimeS-round(.1/options.ControlSampleTimeS))<1e-8, ...
    'ARES:InvalidMapSampleTime','ControlSampleTimeS must divide the 0.1 s LiDAR period.');
assert(abs(options.StopTimeS/.1-round(options.StopTimeS/.1))<1e-8, ...
    'ARES:InvalidMapStopTime','StopTimeS must be a multiple of 0.1 s.');
assert(norm(options.StartState(4:6))<=options.CruiseSpeedMps,'ARES:MapStartSpeed','Start speed exceeds CruiseSpeedMps.');
assert(norm(options.GoalState(4:6))<1e-12,'ARES:MapTerminalVelocity', ...
    'This mission ends in hover: use zero goal velocity.');
assert(all(abs(options.StartEulerZYXRad(2:3))<=25*pi/180), ...
    'ARES:MapInitialAttitude','Initial roll/pitch must be within 25 degrees.');
assert(options.AccelerationLimitMps2<=1.5 && options.JerkLimitMps3<=3, ...
    'ARES:MapAllocationEnvelope','Use acceleration <=1.5 m/s^2 and jerk <=3 m/s^3 for this allocator.');
options.LidarMode=string(options.LidarMode);
assert(isscalar(options.LidarMode) && any(options.LidarMode==["healthy","intermittent","blackout","custom"]), ...
    'ARES:InvalidMapLidarMode','Choose healthy, intermittent, blackout or custom.');
validateattributes(options.OutageDurationS,{'numeric'},{'real','finite','size',[1 2],'positive'});
assert(options.OutageDurationS(2)>=options.OutageDurationS(1),'ARES:InvalidMapDuration','Duration limits must increase.');
validateattributes(options.OutageIntervalsS,{'numeric'},{'real','finite','2d','ncols',2});
end
