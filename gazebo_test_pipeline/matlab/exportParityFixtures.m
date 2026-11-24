function fixturePath = exportParityFixtures(projectRoot)
%EXPORTPARITYFIXTURES Run the MATLAB map-correction functions on fixed test
% inputs and save inputs + outputs as JSON for the Python parity tests.
%
% Usage (from the ARES folder, or with matlab -batch):
%   addpath("gazebo_test_pipeline/matlab");
%   exportParityFixtures();
%
% Reads src/+ares/+navigation and scripts/initializeSingleUav.m; writes only
% gazebo_test_pipeline/tests/fixtures/matlab_parity.json.
arguments
    projectRoot (1,1) string = string(fileparts(fileparts(fileparts(mfilename('fullpath')))))
end
addpath(fullfile(projectRoot,'src'));
rng(7,'twister');
o = ares.navigation.defaultOptions();
o.ShowSimulation = false;
fixture = struct('SchemaVersion',"ares-gz-parity-v1", ...
    'MatlabRelease',string(version('-release')), ...
    'Created',string(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')));
fixture.DefaultOptions = rmfield(o,'ShowSimulation');

% ---- Scene (same path as initializeSimulation -> sceneMap) -------------
init = parseInitializer(fullfile(projectRoot,'scripts','initializeSingleUav.m'));
map = ares.navigation.sceneMap(init.ObstaclePositions,init.ObstacleWidth,init.ObstacleHeight);
fixture.Scene = struct('ObstaclePositions',init.ObstaclePositions, ...
    'ObstacleWidth',init.ObstacleWidth,'ObstacleHeight',init.ObstacleHeight, ...
    'BoxMin',map.BoxMin,'BoxMax',map.BoxMax);

% ---- mapClearance --------------------------------------------------------
n = 200; P = [rand(n,1)*55-10, rand(n,1)*60-10, -rand(n,1)*20];
P(1:12,:) = [10 15 -7; 15 15 -7; 20 10 -7; 18 22 -14.9; 15 5 -0.5; 25 15 -7; ...
    16.4 15 -7; 15 16.4 -7; 13.6 15 -7; 0 0 -7; 30 35 -7; 15 15 -16];
cases = cell(n,1);
for k = 1:n
    [d,nrm] = ares.navigation.mapClearance(P(k,:),map);
    cases{k} = struct('position',P(k,:),'distance',d','normal',nrm);
end
fixture.MapClearance = [cases{:}];

% ---- segmentFree -----------------------------------------------------------
bounds = o.WorldBoundsNedM; cases = cell(n,1);
for k = 1:n
    a = [rand*55-10, rand*60-10, -rand*18-2]; b = [rand*55-10, rand*60-10, -rand*18-2];
    if k <= 20, b = a; end
    pad = rand*2.5;
    cases{k} = struct('first',a,'last',b,'padding',pad, ...
        'free',ares.navigation.segmentFree(a,b,map,pad,bounds));
end
fixture.SegmentFree = [cases{:}];

% ---- projectControl ----------------------------------------------------------
m = 300; cases = cell(m,1);
for k = 1:m
    rows = randi(12); A = randn(rows,3); b = randn(rows,1)*2;
    request = randn(3,1)*2; previous = randn(3,1)*.5;
    lower = max(-1.5,previous-.15); upper = min(1.5,previous+.15);
    if mod(k,3)==0, lower = -1.5*ones(3,1); upper = 1.5*ones(3,1); end
    [u,feasible] = ares.navigation.projectControl(request,A,b,lower,upper);
    cases{k} = struct('request',request','A',A,'b',b','lower',lower','upper',upper', ...
        'solution',u','feasible',feasible);
end
fixture.ProjectControl = [cases{:}];

% ---- correctMotion -----------------------------------------------------------
cases = cell(n,1);
for k = 1:n
    p = [rand*50-8, rand*55-8, -rand*12-4];
    v = randn(1,3)*1.2; nominal = randn(1,3)*1.5; previous = randn(1,3)*.6;
    pts = zeros(0,3);
    if mod(k,4)==0, pts = p + randn(randi(5),3)*3; end
    [u,info] = ares.navigation.correctMotion(p,v,nominal,previous,map,o,pts);
    cases{k} = struct('position',p,'velocity',v,'nominal',nominal,'previous',previous, ...
        'measured',pts,'acceleration',u,'feasible',info.Feasible, ...
        'maximumViolation',info.MaximumViolation,'minimumH',info.MinimumH, ...
        'minimumPsi1',info.MinimumPsi1);
end
fixture.CorrectMotion = [cases{:}];

% ---- planPath --------------------------------------------------------------
missions = {[0 0 -7 0 0 0],[30 35 -7 0 0 0]; [2 -3 -6 .2 0 0],[26 29 -17 0 0 0]; ...
    [0 15 -7 0 0 0],[30 35 -7 0 0 0]; [0 23 -7 0 0 0],[30 31 -7 0 0 0]; ...
    [-5 -5 -8 0 0 0],[40 45 -6 0 0 0]};
cases = cell(size(missions,1),1);
for k = 1:size(missions,1)
    q = o; q.StartState = missions{k,1}; q.GoalState = missions{k,2};
    plan = ares.navigation.planPath(map,q);
    cases{k} = struct('start',q.StartState,'goal',q.GoalState, ...
        'waypoints',plan.WaypointsNedM,'rawWaypoints',plan.RawWaypointsNedM, ...
        'lengthM',plan.LengthM,'expandedNodes',plan.ExpandedNodes);
end
fixture.PlanPath = [cases{:}];

% ---- outageSchedule ------------------------------------------------------------
variants = {struct(), struct('NumberOfOutages',3,'OutageDurationS',[1.5 3], ...
    'InitialHealthyTimeS',3,'FinalHealthyTimeS',4,'MinimumRecoveryTimeS',2,'OutageWindowEndS',21), ...
    struct('NumberOfOutages',1), struct('NumberOfOutages',8,'OutageWindowEndS',40), ...
    struct('LidarMode',"blackout"), struct('LidarMode',"healthy"), ...
    struct('LidarMode',"custom",'OutageIntervalsS',[0 4; 10 15; 21 26])};
cases = {};
for v = 1:numel(variants)
    seeds = 513; if v<=4, seeds = [0:20 513 514 515 99991]; end
    for seed = seeds
        q = o; q.Seed = seed; names = fieldnames(variants{v});
        for f = 1:numel(names), q.(names{f}) = variants{v}.(names{f}); end
        [schedule,intervals] = ares.navigation.outageSchedule(q);
        cases{end+1} = struct('overrides',variants{v},'seed',seed, ...
            'times',double(schedule.Time)','enabled',double(schedule.Data)', ...
            'intervals',intervals); %#ok<AGROW>
    end
end
fixture.OutageSchedule = cases;

% ---- MATLAB RandStream draws (rand and randi) ---------------------------------
s = RandStream('mt19937ar','Seed',513);
draws = struct('seed',513,'rand',rand(s,1,10));
s = RandStream('mt19937ar','Seed',513);
draws.randi = arrayfun(@(imax) randi(s,imax),[1 2 3 7 10 11 20 100 1 5]);
fixture.RandStream = draws;

% ---- virtualLidar -------------------------------------------------------------
poses = [0 0 -7 0 0 0; 12 14 -7 .3 0 0; 16.5 18.5 -7 1.2 .05 -.04; 21 18 -9 -2 0 .1; 30 35 -7 0 0 0];
cases = cell(size(poses,1),1);
for k = 1:size(poses,1)
    [~,ranges,hits] = ares.navigation.virtualLidar(map,poses(k,1:3),poses(k,4:6));
    cases{k} = struct('position',poses(k,1:3),'eulerZYX',poses(k,4:6), ...
        'ranges',ranges','hits',hits');
end
fixture.VirtualLidar = [cases{:}];

% ---- Full controller: a synthetic 6-second track with an unmapped box --------
q = o; q.LidarMode = "custom"; q.OutageIntervalsS = [1 2; 3.5 4];
plan = ares.navigation.planPath(map,q);
folder = tempname; mkdir(folder); planFile = fullfile(folder,'mission.mat');
options = q; save(planFile,'options','map','plan');
controller = ares.navigation.MapCorrectionController('PlanFile',planFile);
% Scan source: the mapped buildings plus one unmapped 1x1x10 m pole.
seen = map; seen.BoxMin(end+1,:) = [4.5 6.5 -10]; seen.BoxMax(end+1,:) = [5.5 7.5 0];
dt = q.ControlSampleTimeS; ticks = 120; trace = cell(ticks,1);
p = q.StartState(1:3); v = [0 0 0]; e = [0 0 0];
for k = 1:ticks
    t = (k-1)*dt;
    enabled = ~any(t>=q.OutageIntervalsS(:,1)-1e-9 & t<q.OutageIntervalsS(:,2)-1e-9);
    points = ares.navigation.virtualLidar(seen,p,e);
    [~,command,packet,stop] = controller(points,p',v',e',double(enabled),t);
    trace{k} = struct('time',t,'position',p,'velocity',v,'eulerZYX',e,'enabled',enabled, ...
        'command',double(command'),'acceleration',packet(5:7)','nominal',packet(8:10)', ...
        'reference',packet(11:13)','feasible',packet(22),'minimumH',packet(24), ...
        'minimumPsi1',packet(25),'goal',packet(26),'waypointIndex',packet(27),'stop',stop);
    % Simple double-integrator stand-in for the plant, only to move the pose.
    u = packet(5:7)'; v = v + u*dt; p = p + v*dt;
    e(1) = e(1) + double(command(3))*dt;
end
fixture.Controller = struct('options',rmfield(q,'ShowSimulation'),'route',plan.WaypointsNedM, ...
    'extraBoxMin',seen.BoxMin(end,:),'extraBoxMax',seen.BoxMax(end,:),'trace',[trace{:}]);
rmdir(folder,'s');

fixturePath = fullfile(projectRoot,'gazebo_test_pipeline','tests','fixtures','matlab_parity.json');
if ~isfolder(fileparts(fixturePath)), mkdir(fileparts(fixturePath)); end
fid = fopen(fixturePath,'w'); assert(fid>=0,'Cannot write %s',fixturePath);
cleanup = onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(fixture));
fprintf('Wrote %s\n',fixturePath);
end

function init = parseInitializer(path)
% Evaluate only the scene constants, without building the uavScenario.
text = fileread(path);
block = regexp(text,'ObstaclePositions\s*=\s*\[.*?\];','match','once');
assert(~isempty(block),'ObstaclePositions not found');
eval(block);
width = regexp(text,'ObstacleWidth\s*=\s*([-+0-9.eE]+);','tokens','once');
height = regexp(text,'ObstacleHeight\s*=\s*([-+0-9.eE]+);','tokens','once');
init = struct('ObstaclePositions',ObstaclePositions, ...
    'ObstacleWidth',str2double(width{1}),'ObstacleHeight',str2double(height{1}));
end
