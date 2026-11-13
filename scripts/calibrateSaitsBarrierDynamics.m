function [calibration, outputDirectory] = calibrateSaitsBarrierDynamics(projectRoot, options)
%CALIBRATESAITSBARRIERDYNAMICS Export the actual guidance plant and its oracles.
% This measures a NEW virtual-acceleration/ZOH software interface. It does not
% claim parity with the existing VFH/PID policy or certify motor hardware.
% Python state order: p_NED, v_NED, roll/pitch/yaw, body_p/q/r, actual thrust.
arguments
    projectRoot (1,1) string = string(fileparts(fileparts(mfilename('fullpath'))))
    options.OutputDirectory (1,1) string = ""
    options.Seed (1,1) double {mustBeInteger,mustBeNonnegative} = 513
    options.PolicyStepS (1,1) double {mustBePositive} = 0.1
    options.CollisionSampleStepS (1,1) double {mustBePositive} = 0.001
    options.TrackingMaxStepS (1,1) double {mustBePositive} = 0.001
    options.EngineeringPaddingMps2 (1,1) double {mustBeNonnegative} = 0.05
end
assert(abs(options.PolicyStepS-0.1)<1e-12 && ...
    options.CollisionSampleStepS<=0.001 && options.TrackingMaxStepS<=0.001 && ...
    abs(options.PolicyStepS/options.CollisionSampleStepS- ...
        round(options.PolicyStepS/options.CollisionSampleStepS))<1e-9, ...
    'ARES:InvalidCalibrationTiming', 'Use the declared 0.1 s policy and <=1 ms truth grid.');
addpath(fullfile(projectRoot,'src'));
cfg = ares.config.defaultSingleUav();
modelName = 'ares_single_uav_top';
modelPath = fullfile(projectRoot,'models',modelName+".slx");
wasLoaded = bdIsLoaded(modelName);
load_system(modelPath);
if ~wasLoaded
    closeModel = onCleanup(@()close_system(modelName,0)); %#ok<NASGU>
end
plant = string(modelName)+"/Quadrotor Plant";
rollPD = numericParameter(plant,'PDRollMultirotor');
pitchPD = numericParameter(plant,'PDPitchMultirotor');
parameters = struct('mass_kg',numericParameter(plant,'MassMultirotor'), ...
    'gravity_mps2',double(cfg.vehicle.gravityMps2), ...
    'roll_kp',rollPD(1),'roll_kd',rollPD(2), ...
    'pitch_kp',pitchPD(1),'pitch_kd',pitchPD(2), ...
    'yaw_rate_gain',numericParameter(plant,'PYawRateMultirotor'), ...
    'thrust_gain',numericParameter(plant,'PThrustMultirotor'), ...
    'roll_limit_rad',25*pi/180,'pitch_limit_rad',25*pi/180, ...
    'yaw_rate_limit_radps',numericParameter(string(modelName)+ ...
    "/Controller/Attitude Control/Saturation",'UpperLimit'));
assert(abs(parameters.mass_kg-cfg.vehicle.massKg)<1e-12, ...
    'ARES:GuidanceMassMismatch', 'Allocator and plant masses differ.');
guidance = configuredModel('double',parameters);
guidanceSingle = configuredModel('single',parameters);
environmentDouble = environment(guidance); environmentDouble.Gravity = parameters.gravity_mps2;
environmentSingle = environment(guidanceSingle); environmentSingle.Gravity = single(parameters.gravity_mps2);
stream = RandStream('mt19937ar','Seed',options.Seed);
outputDirectory = options.OutputDirectory;
if strlength(outputDirectory)==0
    outputDirectory = fullfile(projectRoot,'data','calibration','saits_barrier_dynamics', ...
        string(java.util.UUID.randomUUID()));
end
assert(~isfile(fullfile(outputDirectory,'calibration.json')) && ...
    ~isfile(fullfile(outputDirectory,'fixtures.mat')), 'ARES:CalibrationAlreadyExists', ...
    'Calibration artifacts are immutable; choose a new output directory.');
if ~isfolder(outputDirectory), mkdir(outputDirectory); end

%% Derivative and input-Jacobian oracles from the installed public plant API.
count = 32;
states = zeros(count,13);
states(:,1:3) = [30*rand(stream,count,2),-7*ones(count,1)];
states(:,4:6) = 4*(rand(stream,count,3)-0.5);
states(:,7:8) = 0.4*(rand(stream,count,2)-0.5);
states(:,9) = 2*pi*(rand(stream,count,1)-0.5);
states(:,10:12) = rand(stream,count,3)-0.5;
states(:,13) = parameters.mass_kg*parameters.gravity_mps2*(0.5+rand(stream,count,1));
states(1,:) = [0 0 -7 zeros(1,9) 0];
states(2,:) = states(1,:); states(2,13)=parameters.mass_kg*parameters.gravity_mps2;
commands = [0.4*(rand(stream,count,2)-0.5), ...
    rand(stream,count,1)-0.5, parameters.mass_kg*parameters.gravity_mps2*(0.5+rand(stream,count,1))];
commands(1:2,:) = repmat([0 0 0 parameters.mass_kg*parameters.gravity_mps2],2,1);
states_single = single(states); commands_single = single(commands);
derivatives = zeros(count,13);
derivatives_single = zeros(count,13,'single');
for index = 1:count
    derivatives(index,:) = canonicalDerivative(guidance,environmentDouble,states(index,:),commands(index,:));
    derivatives_single(index,:) = canonicalDerivative(guidanceSingle,environmentSingle, ...
        states_single(index,:),commands_single(index,:));
end
jacobian_case_indices = (1:4)';
control_jacobian = zeros(4,13,4);
jacobianStep = 1e-5;
for row = 1:4
    for column = 1:4
        offset = zeros(1,4); offset(column)=jacobianStep;
        plus = canonicalDerivative(guidance,environmentDouble,states(row,:),commands(row,:)+offset);
        minus = canonicalDerivative(guidance,environmentDouble,states(row,:),commands(row,:)-offset);
        control_jacobian(row,:,column) = (plus-minus)/(2*jacobianStep);
    end
end

%% Short ZOH trajectory reference: tightly integrated authoritative ODE.
rollout_initial_states = states(1:6,:);
rollout_commands = commands(1:6,:);
rollout_time_s = (0:options.CollisionSampleStepS:options.PolicyStepS)';
rollout_trajectory = zeros(6,numel(rollout_time_s),13);
rollout_next_states = zeros(6,13);
referenceOptions = odeset('RelTol',1e-9,'AbsTol',1e-11,'MaxStep',1e-4);
for index = 1:6
    action = rollout_commands(index,:);
    [~,trajectory] = ode15s(@(~,state)canonicalDerivative(guidance,environmentDouble,state,action).', ...
        rollout_time_s,rollout_initial_states(index,:).',referenceOptions);
    rollout_trajectory(index,:,:) = reshape(trajectory,1,numel(rollout_time_s),13);
    rollout_next_states(index,:) = trajectory(end,:);
end
rollout_duration_s = options.PolicyStepS;

%% Allocation reference on the approved software envelope; no post-QP clip.
[aNorth,aEast,aDown,yawGrid] = ndgrid([-1 0 1],[-1 0 1],[-4 0 4],[-pi/2 0 pi/2]);
acceleration_inputs = [aNorth(:),aEast(:),aDown(:)];
allocation_yaw = yawGrid(:);
allocation_yaw_rate = repmat([-.5;0;.5],27,1);
allocation_commands = zeros(size(acceleration_inputs,1),4);
for index = 1:size(acceleration_inputs,1)
    allocation_commands(index,:) = allocate(acceleration_inputs(index,:), ...
        allocation_yaw(index),allocation_yaw_rate(index),parameters);
end
assert(all(abs(allocation_commands(:,1))<parameters.roll_limit_rad) && ...
    all(abs(allocation_commands(:,2))<parameters.pitch_limit_rad), ...
    'ARES:AllocationEnvelopeMismatch','The declared virtual box activates the original attitude clamps.');

%% Empirical tracking domain: ramp from zero virtual acceleration at 1 m/s^3.
% Include all 27 acceleration corners/face-centres/interior points, additional
% yaw-rate cases, and explicit true F=0 startup. The bound is empirical only.
[aNorth,aEast,aDown] = ndgrid([-1 0 1],[-1 0 1],[-4 0 4]);
targets = [aNorth(:),aEast(:),aDown(:)];
targets = [targets;1 1 4;-1 -1 -4;1 -1 4;-1 1 -4;1 1 4;0 0 0];
profileCount = size(targets,1);
yawRate = [zeros(27,1);.5;-.5;.5;-.5;0;0];
initialYaw = [zeros(27,1);pi/2;-pi/2;0;pi;0;0];
startup = false(profileCount,1); startup(end-1:end)=true;
trackingProfiles = repmat(struct('target_acceleration_mps2',[], ...
    'yaw_rate_command_radps',0,'initial_yaw_rad',0,'startup_from_zero_thrust',false, ...
    'maximum_xy_error_mps2',0,'maximum_full3d_error_mps2',0, ...
    'maximum_endpoint_xy_error_mps2',0,'maximum_interval_average_xy_error_mps2',0, ...
    'duration_s',0),profileCount,1);
trackingOptions = odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',options.TrackingMaxStepS);
for profile = 1:profileCount
    state = [0 0 -7 zeros(1,9) parameters.mass_kg*parameters.gravity_mps2];
    state(9) = initialYaw(profile);
    if startup(profile), state(13)=0; end
    desired = [0 0 0];
    totalIntervals = ceil(max(abs(targets(profile,:)))/options.PolicyStepS)+5;
    xyMaximum=0; fullMaximum=0; endpointMaximum=0; averageMaximum=0;
    for interval = 1:totalIntervals
        increment = targets(profile,:)-desired;
        increment = min(max(increment,-options.PolicyStepS),options.PolicyStepS);
        desired = desired+increment;
        action = allocate(desired,state(9),yawRate(profile),parameters);
        time = (0:options.CollisionSampleStepS:options.PolicyStepS)';
        previousVelocity = state(4:6);
        [~,trajectory] = ode15s(@(~,value)canonicalDerivative(guidance,environmentDouble,value,action).', ...
            time,state.',trackingOptions);
        acceleration = zeros(size(trajectory,1),3);
        for sample = 1:size(trajectory,1)
            rate = canonicalDerivative(guidance,environmentDouble,trajectory(sample,:),action);
            acceleration(sample,:) = rate(4:6);
        end
        error = acceleration-desired;
        xy = vecnorm(error(:,1:2),2,2);
        xyMaximum=max(xyMaximum,max(xy)); fullMaximum=max(fullMaximum,max(vecnorm(error,2,2)));
        endpointMaximum=max(endpointMaximum,xy(end));
        actualAverage = (trajectory(end,4:6)-previousVelocity)/options.PolicyStepS;
        averageMaximum=max(averageMaximum,norm(actualAverage(1:2)-desired(1:2)));
        state=trajectory(end,:);
    end
    trackingProfiles(profile) = struct('target_acceleration_mps2',targets(profile,:), ...
        'yaw_rate_command_radps',yawRate(profile),'initial_yaw_rad',initialYaw(profile), ...
        'startup_from_zero_thrust',startup(profile),'maximum_xy_error_mps2',xyMaximum, ...
        'maximum_full3d_error_mps2',fullMaximum, ...
        'maximum_endpoint_xy_error_mps2',endpointMaximum, ...
        'maximum_interval_average_xy_error_mps2',averageMaximum, ...
        'duration_s',totalIntervals*options.PolicyStepS);
    fprintf('DYNAMICS_CALIBRATION profile %d/%d XY error %.6f m/s^2\n', ...
        profile,profileCount,xyMaximum);
end
xyValues = [trackingProfiles.maximum_xy_error_mps2];
fullValues = [trackingProfiles.maximum_full3d_error_mps2];
padding = options.EngineeringPaddingMps2;

%% Save immutable numerical fixtures before publishing the completed JSON.
fixturePath = fullfile(outputDirectory,'fixtures.mat');
save(fixturePath,'states','commands','derivatives','states_single','commands_single', ...
    'derivatives_single','control_jacobian','jacobian_case_indices', ...
    'rollout_initial_states','rollout_commands','rollout_time_s','rollout_trajectory', ...
    'rollout_next_states','rollout_duration_s','acceleration_inputs','allocation_yaw', ...
    'allocation_yaw_rate','allocation_commands','-v7');
calibration.schemaVersion = 'ares-guidance-dynamics-calibration-v1';
calibration.calibrationComplete = true;
calibration.createdAtUtc = string(datetime('now','TimeZone','UTC','Format',"yyyy-MM-dd'T'HH:mm:ss.SSSXXX"));
calibration.parameters = parameters;
calibration.state_order = {'pN','pE','pD','vN','vE','vD','roll','pitch','yaw','body_p','body_q','body_r','actual_thrust'};
calibration.command_order = {'roll','pitch','Euler_yaw_rate','thrust'};
calibration.reference = struct('public_api','multirotor/derivative', ...
    'matlab_release',version('-release'),'platform',computer, ...
    'model_path',modelPath,'model_sha256',ares.inference.fileSha256(modelPath), ...
    'solver_configured',get_param(modelName,'Solver'), ...
    'reference_integrator','ode15s','trajectory_max_step_s',1e-4, ...
    'tracking_max_step_s',options.TrackingMaxStepS, ...
    'closed_loop_guidance_not_motor_hardware',true);
source = fullfile(matlabroot,'toolbox','uav','uavmatlab','+uav','+internal','+impl','+multirotor','GuidanceModel.m');
calibration.reference.derivative_source_path = source;
calibration.reference.derivative_source_sha256 = ares.inference.fileSha256(source);
calibration.reference.calibration_source_sha256 = ares.inference.fileSha256(string(mfilename('fullpath'))+".m");
calibration.software_design_envelope = struct('acceleration_lower_mps2',[-1 -1 -4], ...
    'acceleration_upper_mps2',[1 1 4],'acceleration_slew_mps3',[1 1 1], ...
    'yaw_rate_limit_radps',parameters.yaw_rate_limit_radps, ...
    'roll_limit_rad',parameters.roll_limit_rad,'pitch_limit_rad',parameters.pitch_limit_rad, ...
    'policy_step_s',options.PolicyStepS,'truth_sample_step_s',options.CollisionSampleStepS, ...
    'thrust_lower_n',parameters.mass_kg*(parameters.gravity_mps2-4), ...
    'thrust_upper_n',parameters.mass_kg*(parameters.gravity_mps2+4)/ ...
        (cos(parameters.roll_limit_rad)*cos(parameters.pitch_limit_rad)), ...
    'bounds_are_software_design_not_motor_hardware',true, ...
    'nominal_policy','new frozen differentiable goal/sector policy; original VFH/PID not replayed');
calibration.tracking_error = struct('bound_dimension','horizontal-XY', ...
    'empirical_max_xy_mps2',max(xyValues),'xy_bound_mps2',max(xyValues)+padding, ...
    'bound_mps2',max(xyValues)+padding,'engineering_padding_mps2',padding, ...
    'empirical_max_full3d_mps2',max(fullValues),'full3d_bound_mps2',max(fullValues)+padding, ...
    'startup_max_xy_mps2',max(xyValues(startup)), ...
    'startup_max_full3d_mps2',max(fullValues(startup)), ...
    'flight_ready_xy_bound_mps2',max(xyValues(~startup))+padding, ...
    'max_endpoint_xy_mps2',max([trackingProfiles.maximum_endpoint_xy_error_mps2]), ...
    'max_interval_average_xy_mps2',max([trackingProfiles.maximum_interval_average_xy_error_mps2]), ...
    'policy_step_s',options.PolicyStepS,'profile_count',profileCount, ...
    'provenance','empirical authoritative-ODE grid/ramped-command measurements plus declared engineering padding', ...
    'certified',false,'includes_zero_thrust_startup',true, ...
    'between_grid_times_not_formally_bounded',true);
calibration.tracking_profiles = trackingProfiles;
calibration.tracking_error.coverage = struct('initial_position_ned_m',[0 0 -7], ...
    'initial_velocity_ned_mps',[0 0 0],'initial_roll_pitch_rad',[0 0], ...
    'initial_body_rates_radps',[0 0 0],'virtual_acceleration_starts_at_zero',true, ...
    'slew_limit_is_per_axis',true,'maximum_increment_per_axis_mps2',options.PolicyStepS, ...
    'profile_family','monotone ramps to 27-grid points plus yaw-rate and F=0 startup cases', ...
    'arbitrary_command_histories_not_certified',true);
calibration.internal_state_error_bounds = struct('assumed_perfect_simulator_state',true, ...
    'position_m',0,'velocity_mps',0,'attitude_rad',0,'angular_rate_radps',0,'actual_thrust_n',0, ...
    'hardware_error_bounds_implemented',false);
calibration.hardware_error_bounds = struct('position_m',NaN,'velocity_mps',NaN, ...
    'attitude_rad',NaN,'angular_rate_radps',NaN,'actual_thrust_n',NaN);
calibration.fixtures = struct('file','fixtures.mat','sha256',ares.inference.fileSha256(fixturePath), ...
    'derivative_case_count',count,'control_jacobian_case_count',4, ...
    'rollout_case_count',6,'allocation_case_count',size(acceleration_inputs,1));
writeJson(fullfile(outputDirectory,'calibration.json'),calibration);
fprintf('DYNAMICS_CALIBRATION_COMPLETE %s\n',outputDirectory);
fprintf('Empirical XY bound incl padding: %.6f m/s^2 (full 3D %.6f).\n', ...
    calibration.tracking_error.bound_mps2,calibration.tracking_error.full3d_bound_mps2);
end

function value = numericParameter(block,name)
expression = get_param(block,name);
value = str2num(expression); %#ok<ST2NM> Actual trusted model's numeric mask expressions.
assert(~isempty(value) && all(isfinite(value),'all'), ...
    'ARES:UnresolvedDynamicsParameter','Cannot resolve %s/%s.',block,name);
value = double(value);
end

function model = configuredModel(dataType,parameters)
model = multirotor(dataType);
configuration = model.Configuration;
configuration.PDRoll = cast([parameters.roll_kp parameters.roll_kd],dataType);
configuration.PDPitch = cast([parameters.pitch_kp parameters.pitch_kd],dataType);
configuration.PYawRate = cast(parameters.yaw_rate_gain,dataType);
configuration.PThrust = cast(parameters.thrust_gain,dataType);
configuration.Mass = cast(parameters.mass_kg,dataType);
model.Configuration = configuration;
end

function result = canonicalDerivative(model,environmentValue,state,commands)
order = [1:6 9 8 7 10:13];
dataType = class(model.Configuration.Mass);
controlValue = control(model);
controlValue.Roll = cast(commands(1),dataType);
controlValue.Pitch = cast(commands(2),dataType);
controlValue.YawRate = cast(commands(3),dataType);
controlValue.Thrust = cast(commands(4),dataType);
native = cast(reshape(state,1,13),dataType);
rate = derivative(model,native(order).',controlValue,environmentValue);
result = reshape(rate(order),1,13);
end

function command = allocate(acceleration,yaw,yawRate,parameters)
vertical = parameters.gravity_mps2-acceleration(3);
pitch = atan2(-acceleration(1)*cos(yaw)-acceleration(2)*sin(yaw),vertical);
roll = atan2((-acceleration(1)*sin(yaw)+acceleration(2)*cos(yaw))*cos(pitch),vertical);
thrust = parameters.mass_kg*vertical/(cos(pitch)*cos(roll));
command = [roll pitch yawRate thrust];
end

function writeJson(path,value)
assert(~isfile(path),'ARES:CalibrationAlreadyExists','Calibration JSON already exists.');
file = fopen(path,'w','n','UTF-8');
assert(file~=-1,'ARES:CalibrationWriteFailed','Cannot write calibration JSON.');
cleanup = onCleanup(@()fclose(file)); %#ok<NASGU>
fprintf(file,'%s\n',jsonencode(value,PrettyPrint=true));
end
