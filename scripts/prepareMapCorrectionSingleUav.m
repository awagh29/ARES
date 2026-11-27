%PREPAREMAPCORRECTIONSINGLEUAV Build/compile/open a map-assisted model, no flight.
% Set MapCorrectionOptions first; see docs/MAP_CORRECTION_MATLAB_RUN.md.
mapProjectRoot=string(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(mapProjectRoot,'src'));
if ~exist('MapCorrectionOptions','var'), MapCorrectionOptions=ares.navigation.defaultOptions(); end
mapCorrectionPrepared=ares.navigation.prepareModel(mapProjectRoot,MapCorrectionOptions);
