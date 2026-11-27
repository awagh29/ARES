function wiring = configureModel(model,planFile,options)
%CONFIGUREMODEL Replace executed control on an experiment copy of the plant.
assert(model~="ares_single_uav_top",'ARES:BaselineProtected','Use an experiment copy.');
o=options; dt=num2str(o.ControlSampleTimeS,17);
canonical=["command_roll_rad","command_pitch_rad","command_yaw_rate_radps", ...
    "command_thrust_n","desired_position_ned_m","position_ned_m","velocity_ned_mps"];
ports=find_system(model,'FindAll','on','Type','port','PortType','outport');
for port=ports'
    if string(get_param(port,'DataLogging'))=="on" && any(string(get_param(port,'DataLoggingName'))==canonical)
        set_param(port,'DataLogging','off');
    end
end
state=model+"/ARES Map State";
add_block('simulink/Signal Routing/Bus Selector',state, ...
    'OutputSignals','WorldPosition,WorldVelocity,EulerZYX','Position',[0 650 10 800]);
sp=get_param(state,'PortHandles'); source=get_param(model+"/Rate Transition1",'PortHandles');
add_line(model,source.Outport(1),sp.Inport(1),'autorouting','on');
logPort(sp.Outport(1),'position_ned_m'); logPort(sp.Outport(2),'velocity_ned_mps');
samples=zeros(1,3);
for k=1:3
    path=model+"/ARES Map State Sample "+k;
    add_block('simulink/Discrete/Zero-Order Hold',path,'SampleTime',dt, ...
        'Position',[60 620+55*k 100 645+55*k]);
    p=get_param(path,'PortHandles');
    add_line(model,sp.Outport(k),p.Inport(1),'autorouting','on'); samples(k)=p.Outport(1);
end
available=model+"/ARES Map LiDAR Availability";
add_block('simulink/Sources/From Workspace',available,'VariableName','LidarEnableSchedule', ...
    'Interpolate','off','OutputAfterFinalValue','Holding final value','SampleTime',dt, ...
    'Position',[20 850 160 880]);
clock=model+"/ARES Map Clock";
add_block('simulink/Sources/Digital Clock',clock,'SampleTime',dt,'Position',[40 910 120 940]);
controller=model+"/ARES Map Correction";
add_block('simulink/User-Defined Functions/MATLAB System',controller, ...
    'System','ares.navigation.MapCorrectionController','SimulateUsing','Interpreted execution', ...
    'PlanFile',char(planFile),'Position',[230 650 460 960]);
cp=get_param(controller,'PortHandles'); conversion=get_param(model+"/Conversion",'PortHandles');
add_line(model,conversion.Outport(3),cp.Inport(1),'autorouting','on');
for k=1:3, add_line(model,samples(k),cp.Inport(k+1),'autorouting','on'); end
p=get_param(available,'PortHandles'); add_line(model,p.Outport(1),cp.Inport(5),'autorouting','on');
logPort(p.Outport(1),'map_lidar_available');
p=get_param(clock,'PortHandles'); add_line(model,p.Outport(1),cp.Inport(6),'autorouting','on');
demux=model+"/ARES Map Commands";
add_block('simulink/Signal Routing/Demux',demux,'Outputs','4','Position',[510 650 515 790]);
dp=get_param(demux,'PortHandles'); add_line(model,cp.Outport(2),dp.Inport(1),'autorouting','on');
bus=model+"/ARES Map Control Bus";
add_block('simulink/Signal Routing/Bus Creator',bus,'Inputs','4', ...
    'OutDataTypeStr','Bus: uavGuidanceControlBus','NonVirtualBus','on','Position',[650 650 655 790]);
bp=get_param(bus,'PortHandles'); fields=["Roll","Pitch","YawRate","Thrust"];
for k=1:4
    line=add_line(model,dp.Outport(k),bp.Inport(k),'autorouting','on'); set_param(line,'Name',fields(k));
    logPort(dp.Outport(k),canonical(k));
end
rate=get_param(model+"/Rate Transition2",'PortHandles');
delete_line(get_param(rate.Inport(1),'Line'));
add_line(model,bp.Outport(1),rate.Inport(1),'autorouting','on');
% The original waypoint follower must not terminate a different planned route.
set_param(model+"/Stop Simulation",'Commented','on');
stop=model+"/ARES Map Goal or Failure Stop";
add_block('simulink/Sinks/Stop Simulation',stop,'Position',[600 1060 650 1100]);
p=get_param(stop,'PortHandles'); add_line(model,cp.Outport(4),p.Inport(1),'autorouting','on');
sink=model+"/ARES Map Packet Logger";
add_block('simulink/Sinks/Terminator',sink,'Position',[650 900 670 920]);
p=get_param(sink,'PortHandles'); add_line(model,cp.Outport(3),p.Inport(1),'autorouting','on');
logPort(cp.Outport(3),'map_runtime_packet'); logPort(cp.Outport(4),'map_stop_requested');
% A separate selector logs the actual reference without relabeling old PID data.
select=model+"/ARES Map Reference";
add_block('simulink/Signal Routing/Selector',select,'NumberOfDimensions','1', ...
    'IndexOptionArray',{'Index vector (dialog)'},'IndexParamArray',{'11:13'}, ...
    'InputPortWidth','43','Position',[520 970 575 1000]);
p=get_param(select,'PortHandles'); add_line(model,cp.Outport(3),p.Inport(1),'autorouting','on');
logPort(p.Outport(1),'desired_position_ned_m');
sink=model+"/ARES Map Reference Logger";
add_block('simulink/Sinks/Terminator',sink,'Position',[650 975 670 995]);
q=get_param(sink,'PortHandles'); add_line(model,p.Outport(1),q.Inport(1),'autorouting','on');
set_param(model+"/Quadrotor Plant", ...
    'InitialWorldVelocityMultirotor','MapCorrectionInitialVelocity', ...
    'InitialEulerZYXMultirotor','MapCorrectionInitialEulerZYX', ...
    'InitialBodyAngularRateRPYMultirotor','MapCorrectionInitialBodyRates');
% Avoid storing the full hidden dense scan during blackout. The controller
% ignores it, and this mode records only the selected sector features.
raw=get_param(model+"/UAV Scenario Lidar",'PortHandles');
set_param(raw.Outport(1),'DataLogging','off');
set_param(model,'SignalLogging','on','SignalLoggingName','logsout', ...
    'ReturnWorkspaceOutputs','on','StopTime',num2str(o.StopTimeS),'SimulationMode','normal');
scope=model+"/UAV Scenario Scope";
if o.ShowSimulation
    sp=get_param(scope,'PortHandles'); old=get_param(sp.Inport(1),'Line'); delete_line(old);
    cast=model+"/ARES Map Viewer Cloud";
    add_block('simulink/Signal Attributes/Data Type Conversion',cast, ...
        'OutDataTypeStr','double','Position',[-680 140 -560 170]);
    p=get_param(cast,'PortHandles'); add_line(model,cp.Outport(1),p.Inport(1),'autorouting','on');
    add_line(model,p.Outport(1),sp.Inport(1),'autorouting','on');
    set_param(scope,'Commented','off'); set_param(model,'EnablePacing','on','PacingRate','1');
else
    set_param(scope,'Commented','on'); set_param(model,'EnablePacing','off');
    sink=model+"/ARES Map Hidden Viewer";
    add_block('simulink/Sinks/Terminator',sink,'Position',[650 840 670 860]);
    p=get_param(sink,'PortHandles'); add_line(model,cp.Outport(1),p.Inport(1),'autorouting','on');
end
% Retire dashboard controls which no longer command the executed controller.
delete_block(model+"/Rocker Switch1"); delete_block(model+"/UAV Speed");
old=get_param(model+"/Controller",'PortHandles');
sink=model+"/ARES Unused Original Control";
add_block('simulink/Sinks/Terminator',sink,'Position',[760 530 780 550]);
p=get_param(sink,'PortHandles'); add_line(model,old.Outport(1),p.Inport(1),'autorouting','on');
note=Simulink.Annotation(model,sprintf(['MAP-ASSISTED 3-D NAVIGATION\n' ...
    'Start NED: %s | Goal NED: %s\nLiDAR mode: %s | Requested outages: %d\n' ...
    'Edit MapCorrectionOptions and run prepareMapCorrectionSingleUav to change inputs.'], ...
    mat2str(o.StartState),mat2str(o.GoalState),o.LidarMode,o.NumberOfOutages));
note.Position=[-450 -380];
wiring=struct('ExecutedControlRoute',"ARES Map Correction -> ARES Map Control Bus -> Rate Transition2 -> Quadrotor Plant", ...
    'MapProvidedOnline',true,'UsesML',false,'UsesUnavailableLidar',false, ...
    'SampleTimeS',o.ControlSampleTimeS,'FailurePolicy',"stop simulation on invalid/infeasible filter", ...
    'SafetyCertified',false);
end
function logPort(port,name)
set_param(port,'DataLogging','on','DataLoggingNameMode','Custom','DataLoggingName',name);
end
