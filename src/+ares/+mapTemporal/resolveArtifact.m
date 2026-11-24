function artifact = resolveArtifact(projectRoot,options)
%RESOLVEARTIFACT Export a selected PyTorch checkpoint once; infer in MATLAB.
source=options.ModelPath;
if source==""
    runs=dir(fullfile(projectRoot,'results','sparse_reconstruction_map_temporal','**','best.pt'));
    assert(~isempty(runs),'ARES:TemporalMissingCheckpoint', ...
        'Set MapTemporalOptions.ModelPath to your downloaded best.pt, last.pt, or native model.mat.');
    [~,index]=max([runs.datenum]); source=fullfile(runs(index).folder,runs(index).name);
elseif ~java.io.File(source).isAbsolute()
    source=fullfile(projectRoot,source);
end
if isfolder(source), source=fullfile(source,'best.pt'); end
assert(isfile(source),'ARES:TemporalMissingCheckpoint','Cannot find model: %s',source);
source=string(java.io.File(source).getCanonicalPath());
[~,~,extension]=fileparts(source); manifest=struct();
if strcmpi(extension,'.pt')
    outputRoot=fullfile(projectRoot,'results','map_temporal_exports');
    exporter=fullfile(projectRoot,'scripts','sparsity-invariant-cnn','export_map_temporal_matlab.py');
    % Pass arguments directly to Python, including paths containing spaces.
    args=[options.PythonExecutable,exporter,"--project-root",projectRoot, ...
        "--checkpoint",source,"--output-root",outputRoot];
    command=javaArray('java.lang.String',numel(args));
    for k=1:numel(args), command(k)=java.lang.String(args(k)); end
    builder=java.lang.ProcessBuilder(command); builder.redirectErrorStream(true);
    fprintf('Exporting selected checkpoint to native MATLAB weights...\n');
    try
        process=builder.start(); scanner=java.util.Scanner(process.getInputStream()).useDelimiter('\A');
        output=""; if scanner.hasNext(), output=string(scanner.next()); end
        code=process.waitFor(); scanner.close();
    catch exception
        error('ARES:TemporalPython','Could not start %s. Set PythonExecutable to a Python with torch, numpy and scipy. %s', ...
            options.PythonExecutable,exception.message);
    end
    assert(code==0,'ARES:TemporalExport','Checkpoint export failed:\n%s',output);
    manifest=jsondecode(output); path=string(manifest.model_path);
else
    assert(strcmpi(extension,'.mat'),'ARES:TemporalExtension','Choose a .pt checkpoint or exported model.mat.');
    path=source;
    manifestPath=fullfile(fileparts(path),'export_manifest.json');
    if isfile(manifestPath), manifest=jsondecode(fileread(manifestPath)); end
end
assert(isfile(path),'ARES:TemporalMissingArtifact','Export did not produce model.mat.');
validation=ares.mapTemporal.validateArtifact(path);
if ~isempty(fieldnames(manifest)), manifest.matlab_parity_verified=validation.Passed; end
artifact=struct('SourcePath',source,'SourceSha256',ares.inference.fileSha256(source), ...
    'Path',path,'Sha256',ares.inference.fileSha256(path),'Export',manifest,'Validation',validation);
end
