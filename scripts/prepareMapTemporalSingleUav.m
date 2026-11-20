%PREPAREMAPTEMPORALSINGLEUAV Prepare/open the saved model without running it.
mapTemporalProjectRoot=string(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(mapTemporalProjectRoot,'src'));
if ~exist('MapTemporalOptions','var'), MapTemporalOptions=ares.mapTemporal.defaultOptions(); end
mapTemporalPrepared=ares.mapTemporal.prepareModel(mapTemporalProjectRoot,MapTemporalOptions);
