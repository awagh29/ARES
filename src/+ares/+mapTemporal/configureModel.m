function wiring = configureModel(model,artifactPath,planFile,options)
%CONFIGUREMODEL Keep original guidance/plant; replace missing LiDAR upstream.
wiring=ares.inference.configureSparseReconstructionModel(model,artifactPath);
% The single-episode helper disables VFH memory for its own protocol. This
% temporal experiment retains the baseline controller's original memory.
set_param(model+"/Obstacle Avoidance",'MaxAge',wiring.originalAvoidanceMaxAge);
wiring.experimentAvoidanceMaxAge=str2double(wiring.originalAvoidanceMaxAge);
wiring.AvoidanceMemoryPolicy="Preserve baseline memory for both measured and reconstructed scans";
block=model+"/ARES Sparse Reconstruction";
set_param(block,'System','ares.mapTemporal.LidarReconstruction', ...
    'ArtifactPath',char(artifactPath),'PlanFile',char(planFile));
% Changing System resets mask defaults, including the execution mode.
set_param(block,'SimulateUsing','Interpreted execution');
cp=get_param(block,'PortHandles');
ports=find_system(model,'FindAll','on','Type','port','PortType','outport');
for port=ports'
    name=string(get_param(port,'DataLoggingName'));
    if startsWith(name,'sparse_reconstruction_')
        set_param(port,'DataLoggingName',replace(name,'sparse_reconstruction_','map_temporal_'));
    elseif any(name==["position_ned_m","velocity_ned_mps"])
        set_param(port,'DataLogging','off');
    end
end
state=get_param(model+"/ARES Sparse State",'PortHandles');
logPort(state.Outport(1),'position_ned_m'); logPort(state.Outport(2),'velocity_ned_mps');
% The baseline stops while passing its last waypoint. A zero-velocity goal
% needs a fixed final reference so the original PID can brake into hover.
set_param(model+"/Waypoint Follower",'HoverAtLastWaypoint','on');
selection=get_param(model+"/Enable Obstacle Avoidance",'PortHandles');
controller=get_param(model+"/Controller",'PortHandles');
holdBlock=model+"/ARES Map Temporal Goal Hold";
add_block('simulink/User-Defined Functions/MATLAB System',holdBlock, ...
    'System','ares.mapTemporal.GoalHoldReference','PlanFile',char(planFile), ...
    'SimulateUsing','Interpreted execution','Position',[850 360 1040 470]);
hp=get_param(holdBlock,'PortHandles');
delete_line(model,selection.Outport(1),controller.Inport(2));
delete_line(model,selection.Outport(2),controller.Inport(1));
add_line(model,selection.Outport(1),hp.Inport(1),'autorouting','on');
add_line(model,selection.Outport(2),hp.Inport(2),'autorouting','on');
add_line(model,state.Outport(1),hp.Inport(3),'autorouting','on');
add_line(model,state.Outport(3),hp.Inport(4),'autorouting','on');
add_line(model,hp.Outport(1),controller.Inport(2),'autorouting','on');
add_line(model,hp.Outport(2),controller.Inport(1),'autorouting','on');
set_param(selection.Outport(1),'DataLogging','off');
logPort(hp.Outport(1),'desired_position_ned_m');
logPort(hp.Outport(3),'map_temporal_goal_hold');
sink=model+"/ARES Map Temporal Goal Hold Logger";
add_block('simulink/Sinks/Terminator',sink,'Position',[1110 450 1130 470]);
p=get_param(sink,'PortHandles'); add_line(model,hp.Outport(3),p.Inport(1),'autorouting','on');
set_param(model+"/Stop Simulation",'Commented','on');
stop=model+"/ARES Map Temporal Terminal Stop";
add_block('simulink/Sinks/Stop Simulation',stop,'Position',[630 1080 680 1110]);
p=get_param(stop,'PortHandles'); add_line(model,cp.Outport(15),p.Inport(1),'autorouting','on');
logPort(cp.Outport(15),'map_temporal_terminal_state_reached');
sink=model+"/ARES Map Temporal Runtime Logger";
add_block('simulink/Sinks/Terminator',sink,'Position',[650 1140 670 1160]);
p=get_param(sink,'PortHandles'); add_line(model,cp.Outport(16),p.Inport(1),'autorouting','on');
logPort(cp.Outport(16),'map_temporal_runtime_packet');
% This branch sees the nominal scan for offline error scoring only.
scorer=model+"/ARES Map Temporal Offline Scoring";
add_block('simulink/User-Defined Functions/MATLAB System',scorer, ...
    'System','ares.mapTemporal.NominalRanges','SimulateUsing','Interpreted execution', ...
    'Position',[230 1190 430 1220]);
sp=get_param(scorer,'PortHandles'); raw=get_param(model+"/Conversion",'PortHandles');
add_line(model,raw.Outport(3),sp.Inport(1),'autorouting','on');
sink=model+"/ARES Map Temporal Offline Logger";
add_block('simulink/Sinks/Terminator',sink,'Position',[650 1200 670 1220]);
p=get_param(sink,'PortHandles'); add_line(model,sp.Outport(1),p.Inport(1),'autorouting','on');
logPort(sp.Outport(1),'map_temporal_nominal_ranges_scoring');
raw=get_param(model+"/UAV Scenario Lidar",'PortHandles'); set_param(raw.Outport(1),'DataLogging','off');
set_param(model+"/Quadrotor Plant", ...
    'InitialWorldVelocityMultirotor','MapTemporalInitialVelocity', ...
    'InitialEulerZYXMultirotor','MapTemporalInitialEulerZYX', ...
    'InitialBodyAngularRateRPYMultirotor','MapTemporalInitialBodyRates');
set_param(model,'SignalLogging','on','SignalLoggingName','logsout', ...
    'ReturnWorkspaceOutputs','on','StopTime',num2str(options.StopTimeS),'SimulationMode','normal');
scope=model+"/UAV Scenario Scope";
if options.ShowSimulation
    p=get_param(scope,'PortHandles'); delete_line(get_param(p.Inport(1),'Line'));
    cast=model+"/ARES Map Temporal Viewer Cloud";
    add_block('simulink/Signal Attributes/Data Type Conversion',cast, ...
        'OutDataTypeStr','double','Position',[-680 140 -560 170]);
    q=get_param(cast,'PortHandles'); add_line(model,cp.Outport(1),q.Inport(1),'autorouting','on');
    add_line(model,q.Outport(1),p.Inport(1),'autorouting','on');
    set_param(scope,'Commented','off'); set_param(model,'EnablePacing','on','PacingRate','1');
else
    set_param(scope,'Commented','on'); set_param(model,'EnablePacing','off');
end
note=Simulink.Annotation(model,sprintf(['MAP-TEMPORAL SPARSITY-INVARIANT CNN\n' ...
    'Start NED: %s | Goal NED: %s\nLiDAR: %s | Requested outages: %d\n' ...
    'Known obstacle map + current pose/goal + measured history.\n' ...
    'Original VFH/PID; no barrier correction or global route planner.\n' ...
    'Edit MapTemporalOptions and run prepareMapTemporalSingleUav to change inputs.'], ...
    mat2str(options.StartState),mat2str(options.GoalState),options.LidarMode,options.NumberOfOutages));
note.Position=[-470 -400];
wiring.schemaVersion="ares-map-temporal-controller-wiring-v1";
wiring.ExecutedControlRoute="Selected measured/learned cloud -> original VFH -> original PID -> Quadrotor Plant";
wiring.MapProvidedOnline=true; wiring.UsesML=true; wiring.UsesUnavailableLidar=false;
wiring.History="16 measured-input summaries, states, goals and blackout ages; cleared on every run";
wiring.OfflineScoringRoute="Nominal scan -> ranges -> log terminator only";
wiring.FailurePolicy="Stop on invalid state or nonfinite prediction";
wiring.TerminalPolicy="Within min(0.75 m, endpoint clearance / 2), latch the exact goal reference and current yaw; original PID brakes to hover";
wiring.SafetyCertified=false;
end
function logPort(port,name)
set_param(port,'DataLogging','on','DataLoggingNameMode','Custom','DataLoggingName',name);
end
