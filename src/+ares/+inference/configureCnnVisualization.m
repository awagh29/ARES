function settings = configureCnnVisualization(model, projectRoot, showSimulation, episodeId)
%CONFIGURECNNVISUALIZATION Show the selected controller cloud in live runs.
arguments
    model (1,1) string
    projectRoot (1,1) string
    showSimulation (1,1) logical
    episodeId (1,1) string
end
assert(model~="ares_single_uav_top","ARES:BaselineModelProtected", ...
    "Visualization changes must target the experiment copy.");
scope = model+"/UAV Scenario Scope";
settings.enabled = showSimulation;
settings.cloudSource = "selected-controller-observation";
settings.pacingRate = 1;
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
converter = model+"/ARES CNN Viewer Cloud";
add_block("simulink/Signal Attributes/Data Type Conversion",converter, ...
    "OutDataTypeStr","double","Position",[-640 120 -550 150]);
converterPorts = get_param(converter,"PortHandles");
reconstructionPorts = get_param(model+"/ARES CNN Reconstruction","PortHandles");
add_line(model,reconstructionPorts.Outport(1),converterPorts.Inport(1),"autorouting","on");
add_line(model,converterPorts.Outport(1),scopePorts.Inport(1),"autorouting","on");
% The scope opens its animation during setup. Toolbar Run needs a fresh
% scene and the fault schedule, even after capture restores nominal globals.
quotedRoot = "'"+replace(projectRoot,"'","''")+"'";
quotedEpisode="'"+replace(episodeId,"'","''")+"'";
set_param(model,"InitFcn","ares.inference.prepareCnnSingleEpisodeSimulation("+quotedRoot+","+quotedEpisode+",true);", ...
    "StopFcn","ares.inference.prepareCnnSingleEpisodeSimulation("+quotedRoot+","+quotedEpisode+",false);", ...
    "EnablePacing","on","PacingRate","1");
% Mark only factory-generated wiring clean BEFORE showing it. User edits
% during or after the flight must retain their dirty flag and block replacement.
set_param(model,"Dirty","off");
open_system(model);
drawnow;
fprintf("Live viewer enabled: measured cloud / outage fallback / CNN full predicted scan.\n");
end
