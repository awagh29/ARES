function report = buildVfhMex()
%BUILDVFHMEX Compile aresVfhStep to a MEX and prove it matches the interpreted block.
% Needs MATLAB Coder and a C++ compiler. The MEX is a local build product
% (git-ignored); runGuidanceBridge uses it when present, otherwise the
% interpreted object. Run from any folder:  buildVfhMex
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
addpath(here, fullfile(root, 'src'));
old = cd(here); restore = onCleanup(@() cd(old));
args = {false, zeros(3,1,'single'), zeros(4,1,'single'), zeros(10740,3,'single'), zeros(3,1,'single')};
fprintf('Compiling aresVfhStep_mex ...\n');
codegen('aresVfhStep', '-args', args, '-o', 'aresVfhStep_mex', '-d', fullfile('codegen', 'aresVfhStep'));
clear aresVfhStep aresVfhStep_mex

% Parity: one flight-like sequence through the building cluster, with real scans.
init = parseScene(fullfile(root, 'scripts', 'initializeSingleUav.m'));
map = ares.navigation.sceneMap(init.positions, init.width, init.height);
steps = 60; maxError = 0; statusMatch = true;
for k = 1:steps
    s = (k-1)/(steps-1);
    p = single([0 0 -7] + s*[20 24 0])';
    yaw = .1*sin(k/5);
    q = single(eul2quat([yaw 0 0], 'ZYX'))';
    cloud = ares.navigation.virtualLidar(map, double(p'), [yaw 0 0]);
    target = single([30 35 -7])';
    [d1, y1, s1] = aresVfhStep(k == 1, p, q, cloud, target);
    [d2, y2, s2] = aresVfhStep_mex(k == 1, p, q, cloud, target);
    maxError = max([maxError; abs(double(d1(:)) - double(d2(:))); abs(double(y1) - double(y2))]);
    statusMatch = statusMatch && isequal(s1, s2);
end
report = struct('Steps', steps, 'MaximumDifference', maxError, 'StatusMatches', statusMatch, ...
    'Passed', maxError < 1e-4 && statusMatch);
fprintf('MEX vs interpreted over %d scans: max difference %.3g, status match %d -> %s\n', ...
    steps, maxError, statusMatch, string(report.Passed));
if ~report.Passed
    delete(fullfile(here, ['aresVfhStep_mex.' mexext]));
    error('ARES:VfhMexMismatch', 'Compiled VFH differs from the interpreted block; MEX removed.');
end
end

function init = parseScene(path)
text = fileread(path);
block = regexp(text, 'ObstaclePositions\s*=\s*\[.*?\];', 'match', 'once');
eval(block);
width = regexp(text, 'ObstacleWidth\s*=\s*([-+0-9.eE]+);', 'tokens', 'once');
height = regexp(text, 'ObstacleHeight\s*=\s*([-+0-9.eE]+);', 'tokens', 'once');
init = struct('positions', ObstaclePositions, 'width', str2double(width{1}), 'height', str2double(height{1}));
end
