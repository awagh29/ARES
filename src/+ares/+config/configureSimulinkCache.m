function folders = configureSimulinkCache(projectRoot)
%CONFIGURESIMULINKCACHE Keep generated .slxc files in models, independent of pwd.
% .slxc files are caches, not loadable .slx diagrams. Existing model source
% paths and code-generation settings are intentionally separate.
% Batch workers set ARES_SIMULINK_CACHE_FOLDER for their private cache scope.
arguments
    projectRoot (1,1) string = string(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))))
end
projectRoot=string(java.io.File(projectRoot).getCanonicalPath());
modelFolder=fullfile(projectRoot,'models');
assert(isfile(fullfile(projectRoot,'ares.prj')) && isfolder(modelFolder), ...
    'ARES:InvalidProject','Expected the ARES project and models directory: %s',projectRoot);
cacheFolder=string(getenv('ARES_SIMULINK_CACHE_FOLDER'));
if cacheFolder==""
    cacheFolder=modelFolder;
elseif ~java.io.File(cacheFolder).isAbsolute()
    cacheFolder=fullfile(projectRoot,cacheFolder);
end
cacheFolder=string(java.io.File(cacheFolder).getCanonicalPath());
addpath(modelFolder,'-end');
Simulink.fileGenControl('set','CacheFolder',cacheFolder,'createDir',true);
folders=struct('ModelFolder',modelFolder,'CacheFolder',cacheFolder);
end
