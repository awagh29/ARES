function wiring = configureSaitsModel(model, artifactPath)
%CONFIGURESAITSMODEL Wire only an experiment copy; never persist the baseline.
arguments
    model (1,1) string
    artifactPath (1,1) string
end
assert(model~="ares_single_uav_top","ARES:BaselineModelProtected", ...
    "Instrument an experiment copy, not the baseline model.");
assert(isfile(artifactPath),"ARES:MissingSaitsModel","Missing model artifact.");
cloudBlock = model+"/ARES SAITS Reconstruction";
assert(getSimulinkBlockHandle(cloudBlock)==-1,"ARES:SaitsAlreadyWired", ...
    "The experiment model is already instrumented.");

% Each plant feature is explicitly sampled at the 10-Hz LiDAR rate.
state = model+"/ARES SAITS State";
add_block("simulink/Signal Routing/Bus Selector",state, ...
    "OutputSignals","WorldPosition,WorldVelocity,EulerZYX,BodyAngularRateRPY", ...
    "Position",[0 660 10 810]);
statePorts = get_param(state,"PortHandles");
plantPorts = get_param(model+"/Rate Transition1","PortHandles");
add_line(model,plantPorts.Outport(1),statePorts.Inport(1),"autorouting","on");
sampled = zeros(1,4);
for k = 1:4
    sample = model+"/ARES SAITS State Sample "+k;
    add_block("simulink/Discrete/Zero-Order Hold",sample,"SampleTime","0.1", ...
        "Position",[60 640+40*k 100 665+40*k]);
    ports = get_param(sample,"PortHandles");
    add_line(model,statePorts.Outport(k),ports.Inport(1),"autorouting","on");
    sampled(k) = ports.Outport(1);
end
available = model+"/ARES SAITS Sampled Availability";
add_block("simulink/Sources/From Workspace",available, ...
    "VariableName","LidarEnableSchedule","Interpolate","off", ...
    "OutputAfterFinalValue","Holding final value","SampleTime","0.1", ...
    "Position",[30 870 150 900]);
clock = model+"/ARES SAITS Scan Clock";
add_block("simulink/Sources/Digital Clock",clock,"SampleTime","0.1", ...
    "Position",[40 920 120 950]);
add_block("simulink/User-Defined Functions/MATLAB System",cloudBlock, ...
    "System","ares.inference.SaitsLidarReconstruction", ...
    "SimulateUsing","Interpreted execution","ArtifactPath",char(artifactPath), ...
    "Position",[230 650 440 955]);
ports = get_param(cloudBlock,"PortHandles");
conversionPorts = get_param(model+"/Conversion","PortHandles");
avoidancePorts = get_param(model+"/Obstacle Avoidance","PortHandles");
% Remove the nominal-cloud route before attaching the selected observation.
delete_line(model,conversionPorts.Outport(3),avoidancePorts.Inport(3));
add_line(model,conversionPorts.Outport(3),ports.Inport(1),"autorouting","on");
for k = 1:4
    add_line(model,sampled(k),ports.Inport(k+1),"autorouting","on");
end
p = get_param(available,"PortHandles");
add_line(model,p.Outport(1),ports.Inport(6),"autorouting","on");
p = get_param(clock,"PortHandles");
add_line(model,p.Outport(1),ports.Inport(7),"autorouting","on");
add_line(model,ports.Outport(1),avoidancePorts.Inport(3),"autorouting","on");

names = ["saits_reconstruction_used","saits_history_ready","saits_prediction_valid", ...
    "saits_fallback_reason","saits_inference_seconds","saits_raw_prediction", ...
    "saits_controller_sectors","saits_input_features","saits_input_observed_mask", ...
    "saits_measured_available","saits_backend_mex"];
for k = 1:numel(names)
    sink = model+"/ARES SAITS Log "+k;
    add_block("simulink/Sinks/Terminator",sink,"Position",[530 640+30*k 550 660+30*k]);
    sinkPorts = get_param(sink,"PortHandles");
    add_line(model,ports.Outport(k+1),sinkPorts.Inport(1),"autorouting","on");
    logPort(ports.Outport(k+1),names(k));
end

% Preserve the original native-rate lidar_enabled schedule log and use a
% separate OR to select guidance when current reconstruction is usable.
selection = model+"/Enable Obstacle Avoidance";
reconstructionIn = selection+"/Reconstruction Available";
add_block("simulink/Ports & Subsystems/In1",reconstructionIn,"Port","5", ...
    "Position",[-135 380 -105 400]);
gate = selection+"/ARES Guidance Available";
add_block("simulink/Logic and Bit Operations/Logical Operator",gate, ...
    "Operator","OR","Inputs","2","Position",[35 260 75 310]);
reconPorts = get_param(reconstructionIn,"PortHandles");
gatePorts = get_param(gate,"PortHandles");
schedulePorts = get_param(selection+"/Enable1","PortHandles");
switchPaths = [selection+"/Switch",selection+"/Switch1"];
for path = switchPaths
    switchPorts = get_param(path,"PortHandles");
    delete_line(selection,schedulePorts.Outport(1),switchPorts.Inport(2));
    add_line(selection,gatePorts.Outport(1),switchPorts.Inport(2),"autorouting","on");
end
add_line(selection,schedulePorts.Outport(1),gatePorts.Inport(1),"autorouting","on");
add_line(selection,reconPorts.Outport(1),gatePorts.Inport(2),"autorouting","on");
logPort(schedulePorts.Outport(1),"lidar_enabled");
logPort(gatePorts.Outport(1),"saits_guidance_enabled");
selectionPorts = get_param(selection,"PortHandles");
add_line(model,ports.Outport(2),selectionPorts.Inport(5),"autorouting","on");

% Do not retain a synthetic proxy cloud after returning to a real scan.
% The model still has its own 50-scan history. This setting is recorded.
wiring.originalAvoidanceMaxAge = string(get_param(model+"/Obstacle Avoidance","MaxAge"));
set_param(model+"/Obstacle Avoidance","MaxAge","0");
wiring.experimentAvoidanceMaxAge = 0;

% Reference proof: actual selected guidance and the same-state waypoint
% alternative, upstream of the existing PID and four executed commands.
wf = get_param(selection+"/wf_desiredPosition","PortHandles");
selected = get_param(selection+"/Switch","PortHandles");
logPort(wf.Outport(1),"saits_waypoint_reference_ned_m");
logPort(selected.Outport(1),"saits_selected_reference_ned_m");
delta = selection+"/ARES Reference Difference";
add_block("simulink/Math Operations/Sum",delta,"Inputs","+-", ...
    "Position",[400 420 430 460]);
deltaPorts = get_param(delta,"PortHandles");
add_line(selection,selected.Outport(1),deltaPorts.Inport(1),"autorouting","on");
add_line(selection,wf.Outport(1),deltaPorts.Inport(2),"autorouting","on");
sink = selection+"/ARES Reference Difference Logger";
add_block("simulink/Sinks/Terminator",sink,"Position",[510 430 530 450]);
p = get_param(sink,"PortHandles");
add_line(selection,deltaPorts.Outport(1),p.Inport(1),"autorouting","on");
logPort(deltaPorts.Outport(1),"saits_reference_delta_ned_m");
% Preserve canonical desired_position_ned_m at the subsystem boundary.
logPort(selectionPorts.Outport(1),"desired_position_ned_m");
logPort(sampled(3),"saits_attitude_zyx_rad");
logPort(sampled(4),"saits_angular_velocity_body_radps");
wiring.schemaVersion = "ares-saits-controller-wiring-v1";
wiring.artifactSha256 = ares.inference.fileSha256(artifactPath);
wiring.pointCloudAdapter = "forward-sector-conservative-proxy-with-unknown-side-rear-guard";
wiring.reconstructionSampleTimeSeconds = .1;
wiring.originalModelUnchanged = true;
end

function logPort(port, name)
set_param(port,"DataLogging","on","DataLoggingNameMode","Custom","DataLoggingName",name);
end
