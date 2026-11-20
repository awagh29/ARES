%RUNMAPTEMPORALSINGLEUAV Select checkpoint, start/goal and LiDAR outage mode.
mapTemporalProjectRoot=string(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(mapTemporalProjectRoot,'src'));
if ~exist('MapTemporalOptions','var'), MapTemporalOptions=ares.mapTemporal.defaultOptions(); end
mapTemporalPrepared=ares.mapTemporal.prepareModel(mapTemporalProjectRoot,MapTemporalOptions);
mapTemporalResult=ares.mapTemporal.runMission(mapTemporalPrepared);
