function settings = configureDenseSaitsVisualization(model, projectRoot, showSimulation, episodeId, lidarMode)
%CONFIGUREDENSESAITSVISUALIZATION Display measured or dense reconstructed clouds during live runs.
arguments
    model (1,1) string
    projectRoot (1,1) string
    showSimulation (1,1) logical
    episodeId (1,1) string
    lidarMode (1,1) string {mustBeMember(lidarMode,["intermittent","disabled"])} = "intermittent"
end
assert(model~="ares_single_uav_top","ARES:BaselineModelProtected", ...
    "Visualization changes must target the experiment copy.");
scope = model+"/UAV Scenario Scope";
settings.enabled = showSimulation;
settings.cloudSource = "selected-controller-observation";
settings.pacingRate = 1;
settings.lidarMode = lidarMode;
if ~showSimulation
    set_param(scope,"Commented","on");
    set_param(model,"EnablePacing","off");
    return;
end
set_param(scope,"Commented","off");
scopePorts = get_param(scope,"PortHandles");
oldLine = get_param(scopePorts.Inport(1),"Line");
oldSource = get_param(oldLine,"SrcPortHandle");
delete_line(model,oldSource,scopePorts.Inport(1));
converter = model+"/ARES DENSE SAITS Viewer Cloud";
add_block("simulink/Signal Attributes/Data Type Conversion",converter, ...
    "OutDataTypeStr","double","Position",[-640 120 -550 150]);
converterPorts = get_param(converter,"PortHandles");
reconstructionPorts = get_param(model+"/ARES DENSE SAITS Reconstruction","PortHandles");
add_line(model,reconstructionPorts.Outport(1),converterPorts.Inport(1),"autorouting","on");
add_line(model,converterPorts.Outport(1),scopePorts.Inport(1),"autorouting","on");
% The scope opens its animation during setup. Toolbar Run needs a fresh
% scene and the fault schedule, even after capture restores nominal globals.
quotedRoot = "'"+replace(projectRoot,"'","''")+"'";
quotedEpisode="'"+replace(episodeId,"'","''")+"'";
quotedMode="'"+lidarMode+"'";
set_param(model,"InitFcn","ares.inference.prepareDenseSaitsSingleEpisodeSimulation("+quotedRoot+","+quotedEpisode+",true,"+quotedMode+");", ...
    "StopFcn","ares.inference.prepareDenseSaitsSingleEpisodeSimulation("+quotedRoot+","+quotedEpisode+",false,"+quotedMode+");", ...
    "EnablePacing","on","PacingRate","1");
% Mark only factory-generated wiring clean BEFORE showing it. User edits
% during or after the flight must retain their dirty flag and block replacement.
set_param(model,"Dirty","off");
open_system(model);
drawnow;
fprintf("Live viewer enabled: measured cloud / outage fallback / DENSE SAITS full predicted scan.\n");
end
