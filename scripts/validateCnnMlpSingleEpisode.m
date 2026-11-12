function report = validateCnnMlpSingleEpisode(projectRoot,episodeId)
%VALIDATECNNMLPSINGLEEPISODE Native parity, source isolation, and existing plant wiring.
arguments
    projectRoot (1,1) string
    episodeId (1,1) string
end
addpath(fullfile(projectRoot,"src"));
deployment=jsondecode(fileread(fullfile(projectRoot,"data","models", ...
    "cnn_mlp_single_episode",episodeId,"deployment.json")));
artifact=string(deployment.model_path);
report=ares.inference.validateCnnMlp(artifact);
[cfg,source]=ares.inference.cnnSingleEpisodeConfig(projectRoot,episodeId);
assert(isequal(cfg.mission.initialPositionNed,reshape(double(source.configuration.mission.initialPositionNed),1,3)), ...
    "ARES:CnnStartMismatch","Start was not restored from the selected episode.");
baselineCfg=ares.config.defaultSingleUav();
comparison=cfg;
comparison.mission.initialPositionNed=baselineCfg.mission.initialPositionNed;
comparison.mission.waypointsNed(1,:)=baselineCfg.mission.waypointsNed(1,:);
comparison.sensors=baselineCfg.sensors;
assert(isequaln(comparison,baselineCfg),"ARES:CnnFrameworkChanged","Only the start and sensor mode may vary.");
a=ares.inference.CnnMlpLidarReconstruction("ArtifactPath",char(artifact));
b=ares.inference.CnnMlpLidarReconstruction("ArtifactPath",char(artifact));
cleanup=onCleanup(@()releaseBoth(a,b)); %#ok<NASGU>
p=cfg.mission.initialPositionNed; v=[.1 .2 0]; euler=[.3 .1 -.1]; omega=[0 0 0];
oracleA=ones(10740,3,"single"); oracleB=NaN(10740,3,"single");
[cloudA,usedA,readyA,validA,~,~,~,~,featuresA,maskA,availableA,~,denseA]=a(oracleA,p,v,euler,omega,false,0);
[cloudB,usedB,~,~,~,~,~,~,featuresB,maskB,~,~,denseB]=b(oracleB,p,v,euler,omega,false,0);
assert(usedA && usedB && readyA && validA && ~availableA && all(maskA) && all(maskB), ...
    "ARES:CnnOutageInferenceFailed","State-only inference should work on the first outage scan.");
assert(isequaln(cloudA,cloudB) && isequal(denseA,denseB) && isequal(featuresA,featuresB), ...
    "ARES:CnnOracleLeak","Outage oracle points affected the CNN output.");
expected=single([p v euler([3 2 1])]);
angles=expected(7:9);
expected=[expected(1:6),reshape([sin(angles);cos(angles)],1,6)].';
assert(max(abs(featuresA-expected))<1e-6,"ARES:CnnStateOrder","Incorrect state or angle feature order.");
[repeated,repeatedUsed]=a(oracleB,p+[100 0 0],v,euler,omega,true,0);
assert(repeatedUsed && isequaln(repeated,cloudA),"ARES:CnnRepeatedTick","Same-tick state must remain cached.");
[healthy,usedHealthy,~,~,~,~,~,~,~,~,~,~,unusedDense]=a(oracleA,p,v,euler,omega,true,.1);
assert(~usedHealthy && isequaln(healthy,oracleA),"ARES:CnnRecoveryFailed","Healthy scans must pass through unchanged.");
assert(all(unusedDense==0),"ARES:CnnDenseLogFill","Unused dense log must be finite zero placeholders.");
report.startPositionNed=cfg.mission.initialPositionNed;
report.outageOracleIsolationPassed=true;
report.stateFeatureOrderPassed=true;
report.healthyRecoveryPassed=true;
report.repeatedTickPassed=true;
path=fullfile(fileparts(artifact),"integration_validation.json");
fid=fopen(path,"w","n","UTF-8"); assert(fid~=-1,"ARES:WriteFailed","Cannot write validation.");
fileCleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
disp(report);
end

function releaseBoth(a,b)
release(a); release(b);
end
