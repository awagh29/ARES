%RUNMAPCORRECTIONSINGLEUAV Prepare and run with user-selected start/goal/outages.
mapProjectRoot=string(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(mapProjectRoot,'src'));
if ~exist('MapCorrectionOptions','var'), MapCorrectionOptions=ares.navigation.defaultOptions(); end
mapCorrectionPrepared=ares.navigation.prepareModel(mapProjectRoot,MapCorrectionOptions);
mapCorrectionResult=ares.navigation.runMission(mapCorrectionPrepared);
