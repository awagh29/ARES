function configureGovernor(model,planFile,cloudPort,approachPort)
%CONFIGUREGOVERNOR Insert between position and velocity PID in experiment copy.
assert(model~="ares_single_uav_top",'ARES:BaselineModelProtected','Use an experiment copy.');
controller=model+"/Controller";
for k=1:2
    name=["Governor Cloud","Governor Approach"];
    b=controller+"/"+name(k);
    add_block('simulink/Ports & Subsystems/In1',b,'Port',num2str(k+3), ...
        'Position',[-300 370+60*k -270 390+60*k]);
end
cp=get_param(controller,'PortHandles');
add_line(model,cloudPort,cp.Inport(4),'autorouting','on');
add_line(model,approachPort,cp.Inport(5),'autorouting','on');
state=controller+"/Governor State";
add_block('simulink/Signal Routing/Bus Selector',state, ...
    'OutputSignals','WorldPosition,WorldVelocity,EulerZYX','Position',[-100 400 -90 490]);
sp=get_param(state,'PortHandles'); p=get_param(controller+"/UAV State",'PortHandles');
add_line(controller,p.Outport(1),sp.Inport(1),'autorouting','on');
governor=controller+"/ARES Braking Speed Governor";
add_block('simulink/User-Defined Functions/MATLAB System',governor, ...
    'System','ares.mapTemporal.SpeedGovernor','PlanFile',char(planFile), ...
    'SimulateUsing','Interpreted execution','Position',[180 400 410 610]);
gp=get_param(governor,'PortHandles');
position=get_param(controller+"/Position Control",'PortHandles');
velocity=get_param(controller+"/Velocity Control",'PortHandles');
delete_line(controller,position.Outport(1),velocity.Inport(1));
add_line(controller,position.Outport(1),gp.Inport(1),'autorouting','on');
for k=1:3, add_line(controller,sp.Outport(k),gp.Inport(k+1),'autorouting','on'); end
for k=1:2
    name=["Governor Cloud","Governor Approach"]; p=get_param(controller+"/"+name(k),'PortHandles');
    add_line(controller,p.Outport(1),gp.Inport(k+4),'autorouting','on');
end
add_line(controller,gp.Outport(1),velocity.Inport(1),'autorouting','on');
% Baseline outer-loop integral gains are zero (Ix/Iy/Iz in initializer).
% Preserve those PID blocks and gains; there is no outer integral to wind up.
sink=controller+"/Governor Diagnostics";
add_block('simulink/Sinks/Terminator',sink,'Position',[490 580 510 600]);
p=get_param(sink,'PortHandles'); add_line(controller,gp.Outport(2),p.Inport(1),'autorouting','on');
set_param(gp.Outport(1),'DataLogging','on','DataLoggingNameMode','Custom', ...
    'DataLoggingName','map_temporal_governed_velocity_ned_mps');
set_param(gp.Outport(2),'DataLogging','on','DataLoggingNameMode','Custom', ...
    'DataLoggingName','map_temporal_governor_packet');
add_block('simulink/Sinks/Terminator',controller+"/Planner Diagnostics",'Position',[490 630 510 650]);
p=get_param(controller+"/Planner Diagnostics",'PortHandles');
add_line(controller,gp.Outport(3),p.Inport(1),'autorouting','on');
set_param(gp.Outport(3),'DataLogging','on','DataLoggingNameMode','Custom', ...
    'DataLoggingName','map_temporal_planner_packet');
end
