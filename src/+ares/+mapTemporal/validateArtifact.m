function report = validateArtifact(artifactPath)
%VALIDATEARTIFACT Verify the actual native CNN/GRU against PyTorch fixtures.
data=load(artifactPath,'model','fixtures','absoluteTolerance');
assert(string(data.model.schemaVersion)=="ares-map-temporal-native-v1",'ARES:TemporalSchema','Wrong model artifact.');
count=numel(data.fixtures); errors=zeros(count,1); times=errors;
for k=1:count
    f=data.fixtures{k};
    buffer=struct('States',single(f.states),'Goals',single(f.goals),'Summaries',single(f.history), ...
        'Present',logical(f.present),'Ages',single(f.ages));
    timer=tic;
    predicted=ares.mapTemporal.predict(data.model,f.ranges,logical(f.valid),logical(f.available),buffer);
    times(k)=toc(timer); errors(k)=max(abs(double(predicted)-double(f.expected)),[],'all');
end
report=struct('SchemaVersion',"ares-map-temporal-parity-v1", ...
    'ModelSha256',ares.inference.fileSha256(artifactPath),'FixtureCount',count, ...
    'MaximumErrorM',max(errors),'AbsoluteToleranceM',double(data.absoluteTolerance), ...
    'MedianInferenceSeconds',median(times),'PerFixtureErrorsM',errors, ...
    'Passed',all(errors<=data.absoluteTolerance),'MatlabRelease',string(version('-release')));
assert(report.Passed,'ARES:TemporalParity','MATLAB/PyTorch inference differs by %.8g m (limit %.8g).',max(errors),data.absoluteTolerance);
path=fullfile(fileparts(artifactPath),'matlab_validation.json');
fid=fopen(path,'w'); assert(fid>=0,'ARES:TemporalWrite','Cannot write validation report.');
cleanup=onCleanup(@()fclose(fid)); fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true)); %#ok<NASGU>
fprintf('Map-temporal native parity: %.3g m maximum error; median inference %.3f s.\n',max(errors),median(times));
end
