function report = validateSparseReconstructionSingleEpisode(projectRoot,episodeId)
%VALIDATESPARSERECONSTRUCTIONSINGLEEPISODE Numerical checks and compile, no flight.
arguments
    projectRoot (1,1) string
    episodeId (1,1) string
end
addpath(fullfile(projectRoot,"src"));
pointer=jsondecode(fileread(fullfile(projectRoot,"data","models", ...
    "sparse_reconstruction_single_episode",episodeId,"deployment.json")));
artifact=string(pointer.model_path);
report=ares.inference.validateSparseReconstruction(artifact);
results=runtests(fullfile(projectRoot,"tests","data","TestSparseReconstructionScenario.m"));
assert(all([results.Passed]),"ARES:SparseScenarioTestsFailed","Sparse scenario checks failed or were skipped.");
report.scenarioTestCount=numel(results);
baseline=fullfile(projectRoot,"models","ares_single_uav_top.slx");
baselineHash=ares.inference.fileSha256(baseline);
report.compiledModes=strings(0,1);
for mode=["intermittent","disabled"]
    model="ares_sparse_compile_"+mode;
    assert(~bdIsLoaded(model),"ARES:SparseValidationModelOpen","Close %s before validation.",model);
    folder=fullfile(projectRoot,"results","sparse_reconstruction_validation",episodeId);
    if ~isfolder(folder), mkdir(folder); end
    copy=fullfile(folder,model+".slx");
    if isfile(copy)
        assert(ares.inference.fileSha256(copy)==baselineHash, ...
            "ARES:SparseValidationCopyChanged","Preserve changed validation copy before retrying: %s",copy);
    else
        copyfile(baseline,copy);
    end
    cleanup=onCleanup(@()cleanupModel(model,projectRoot,episodeId,mode));
    ares.inference.prepareSparseReconstructionSingleEpisodeSimulation(projectRoot,episodeId,true,mode);
    load_system(copy);
    ares.inference.configureSparseReconstructionModel(model,artifact);
    ares.inference.configureSparseReconstructionVisualization(model,projectRoot,false,episodeId,mode);
    quotedRoot="'"+replace(projectRoot,"'","''")+"'";
    quotedEpisode="'"+replace(episodeId,"'","''")+"'";
    set_param(model,"InitFcn","ares.inference.prepareSparseReconstructionSingleEpisodeSimulation("+ ...
        quotedRoot+","+quotedEpisode+",true,'"+mode+"');", ...
        "StopFcn","","SimulationCommand","update");
    report.compiledModes(end+1)=mode;
    clear cleanup;
end
assert(ares.inference.fileSha256(baseline)==baselineHash,"ARES:BaselineChanged","Baseline file changed.");
report.originalModelUnchanged=true;
report.plantSimulationRun=false;
path=fullfile(folder,"integration_validation.json");
fid=fopen(path,"w","n","UTF-8"); assert(fid~=-1,"ARES:WriteFailed","Cannot write validation.");
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
disp(report);
end

function cleanupModel(model,root,episode,mode)
if bdIsLoaded(model), close_system(model,0); end
ares.inference.prepareSparseReconstructionSingleEpisodeSimulation(root,episode,false,mode);
end
