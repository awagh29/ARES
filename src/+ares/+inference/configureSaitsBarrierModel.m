function wiring = configureSaitsBarrierModel(model,projectRoot,deploymentPath,pythonExecutable,cfg,continueAfterFailure)
%CONFIGURESAITSBARRIERMODEL Execute the trained controller, not a VFH proxy.
assert(model~="ares_single_uav_top","ARES:BaselineProtected","Use an experiment copy.");
state = model+"/ARES Barrier State";
add_block("simulink/Signal Routing/Bus Selector",state, ...
    "OutputSignals","WorldPosition,WorldVelocity,EulerZYX,BodyAngularRateRPY", ...
    "Position",[0 660 10 810]);
sp = get_param(state,"PortHandles");
plant = get_param(model+"/Rate Transition1","PortHandles");
add_line(model,plant.Outport(1),sp.Inport(1),"autorouting","on");
sampled = zeros(1,4);
for k=1:4
    block = model+"/ARES Barrier State Sample "+k;
    add_block("simulink/Discrete/Zero-Order Hold",block,"SampleTime","0.1", ...
        "Position",[60 640+40*k 100 665+40*k]);
    p = get_param(block,"PortHandles");
    add_line(model,sp.Outport(k),p.Inport(1),"autorouting","on"); sampled(k)=p.Outport(1);
end
available = model+"/ARES Barrier Availability";
add_block("simulink/Sources/From Workspace",available,"VariableName","LidarEnableSchedule", ...
    "Interpolate","off","OutputAfterFinalValue","Holding final value","SampleTime","0.1", ...
    "Position",[30 870 150 900]);
clock = model+"/ARES Barrier Clock";
add_block("simulink/Sources/Digital Clock",clock,"SampleTime","0.1","Position",[40 920 120 950]);
controller = model+"/ARES SAITS Reconstruction";
add_block("simulink/User-Defined Functions/MATLAB System",controller, ...
    "System","ares.inference.SaitsBarrierController","SimulateUsing","Interpreted execution", ...
    "ProjectRoot",char(projectRoot),"DeploymentPath",char(deploymentPath), ...
    "PythonExecutable",char(pythonExecutable),"GoalNed",mat2str(cfg.mission.waypointsNed(end,:)), ...
    "ContinueAfterFailure",mat2str(logical(continueAfterFailure)), ...
    "Position",[230 650 450 950]);
cp = get_param(controller,"PortHandles");
conversion = get_param(model+"/Conversion","PortHandles");
add_line(model,conversion.Outport(3),cp.Inport(1),"autorouting","on");
for k=1:4, add_line(model,sampled(k),cp.Inport(k+1),"autorouting","on"); end
p=get_param(available,"PortHandles"); add_line(model,p.Outport(1),cp.Inport(6),"autorouting","on");
logPort(p.Outport(1),"barrier_measured_available");
p=get_param(clock,"PortHandles"); add_line(model,p.Outport(1),cp.Inport(7),"autorouting","on");

% Remove old canonical command/reference log names, so telemetry denotes the
% signals actually executed, rather than the now-bypassed nominal PID.
ports=find_system(model,'FindAll','on','Type','port','PortType','outport');
names=["command_roll_rad","command_pitch_rad","command_yaw_rate_radps", ...
    "command_thrust_n","desired_position_ned_m"];
for port=ports'
    if string(get_param(port,"DataLogging"))=="on" && any(string(get_param(port,"DataLoggingName"))==names)
        set_param(port,"DataLogging","off");
    end
end
demux=model+"/ARES Barrier Commands";
add_block("simulink/Signal Routing/Demux",demux,"Outputs","4","Position",[510 650 515 790]);
dp=get_param(demux,"PortHandles"); add_line(model,cp.Outport(2),dp.Inport(1),"autorouting","on");
bus=model+"/ARES Barrier Control Bus";
add_block("simulink/Signal Routing/Bus Creator",bus,"Inputs","4", ...
    "OutDataTypeStr","Bus: uavGuidanceControlBus","NonVirtualBus","on", ...
    "Position",[620 650 625 790]);
bp=get_param(bus,"PortHandles");
for k=1:4
    line=add_line(model,dp.Outport(k),bp.Inport(k),"autorouting","on");
    fields=["Roll","Pitch","YawRate","Thrust"]; set_param(line,"Name",fields(k));
    logPort(dp.Outport(k),names(k));
end
rate=get_param(model+"/Rate Transition2","PortHandles");
old=get_param(rate.Inport(1),"Line"); delete_line(old);
add_line(model,bp.Outport(1),rate.Inport(1),"autorouting","on");

split=model+"/ARES Barrier Diagnostics";
add_block("simulink/Signal Routing/Demux",split,"Outputs","[4 14 14 31 31 3 3 3 10]", ...
    "Position",[510 830 515 1100]);
p=get_param(split,"PortHandles"); add_line(model,cp.Outport(3),p.Inport(1),"autorouting","on");
logNames=["barrier_executed_commands","barrier_prediction","barrier_perceived_lidar", ...
    "barrier_input_features","barrier_input_mask","desired_position_ned_m", ...
    "barrier_filtered_acceleration","barrier_nominal_acceleration","barrier_diagnostics"];
for k=1:9
    sink=model+"/ARES Barrier Log "+k;
    add_block("simulink/Sinks/Terminator",sink,"Position",[720 810+30*k 740 830+30*k]);
    sinkPort=get_param(sink,"PortHandles");
    add_line(model,p.Outport(k),sinkPort.Inport(1),"autorouting","on"); logPort(p.Outport(k),logNames(k));
end
stop=model+"/ARES Barrier Failure Stop";
add_block("simulink/Sinks/Stop Simulation",stop,"Position",[550 1150 590 1180]);
p=get_param(stop,"PortHandles"); add_line(model,cp.Outport(4),p.Inport(1),"autorouting","on");
logPort(cp.Outport(4),"barrier_failure_stop");
logPort(cp.Outport(3),"barrier_runtime_packet");
wiring.schemaVersion="ares-augmented-saits-barrier-wiring-v1";
wiring.controller="state-augmented SAITS + frozen reference/controller + learned-gain hard QP + exact allocation";
wiring.executedControlRoute="ARES Barrier Control Bus -> Rate Transition2 -> Quadrotor Plant";
wiring.previousControllerBypassed=true;
wiring.failurePolicy="Stop Simulation at first invalid/infeasible safety filter; bounded emergency output is not certified safe";
if continueAfterFailure
    wiring.failurePolicy="Diagnostic continuation: bounded nominal perceived-LiDAR control on invalid/infeasible QP; uncertified";
end
wiring.continueAfterBarrierFailure=logical(continueAfterFailure);
wiring.filterFailureStopsSimulation=~logical(continueAfterFailure);
wiring.samplePeriodS=.1;
wiring.backend="persistent out-of-process Python; CPU FP32 neural and FP64 QP; no MEX";
wiring.oracleMapInput=false;
wiring.diagnosticOrder=["available","reconstructionUsed","feasible","maxViolation", ...
    "trackingSlack","k2","failureLatched","computeSeconds","minimumH","minimumPsi1"];
end

function logPort(port,name)
set_param(port,"DataLogging","on","DataLoggingNameMode","Custom","DataLoggingName",name);
end
