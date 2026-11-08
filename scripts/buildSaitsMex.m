function build = buildSaitsMex(outputDirectory)
%BUILDSAITSMEX Compile native SAITS C++ using MATLAB BLAS, without MATLAB Coder.
% Entry: [physical, normalized] = saits_endpoint_mex(model, rawSingle, mask).
% The exported model is an explicit argument, so the binary never caches or
% silently retains another checkpoint's parameters. validateSaitsMatlab builds
% and verifies this entry before saving its artifact/hash manifest.
arguments
    outputDirectory (1,1) string = ""
end
projectRoot = string(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(projectRoot, 'src'));
if strlength(outputDirectory) == 0
    outputDirectory = fullfile(projectRoot, 'data', 'models', 'saits_endpoint', 'mex');
end
source = fullfile(projectRoot, 'src', '+ares', '+inference', 'saits_endpoint_mex.cpp');
assert(isfile(source), 'ARES:MissingSaitsCppSource', 'Native SAITS C++ source is missing.');
compiler = mex.getCompilerConfigurations('C++', 'Selected');
assert(~isempty(compiler), 'ARES:MissingCppCompiler', ...
    'Select a supported C++ compiler with mex -setup C++ before building SAITS.');
if ~isfolder(outputDirectory)
    mkdir(outputDirectory);
end
clear saits_endpoint_mex;
mex('-R2018a', '-O', char(source), '-lmwblas', ...
    '-outdir', char(outputDirectory), '-output', 'saits_endpoint_mex');
binary = fullfile(outputDirectory, "saits_endpoint_mex." + string(mexext));
assert(isfile(binary), 'ARES:MissingSaitsMex', 'Native SAITS MEX compilation did not produce its binary.');
addpath(outputDirectory);
clear saits_endpoint_mex;
build.backend = 'cpp-matlab-blas';
build.mexFullPath = binary;
build.mexFunctionName = 'saits_endpoint_mex';
build.mexIncludesModelArgument = true;
build.sourcePath = source;
build.sourceSha256 = ares.inference.fileSha256(source);
build.builderSourceSha256 = ares.inference.fileSha256(mfilename('fullpath') + ".m");
build.compilerName = string(compiler(1).Name);
build.requiresMatlabCoder = false;
end
