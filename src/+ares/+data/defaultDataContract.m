function contract = defaultDataContract()
% For one UAV ONLY!!
contract.schemaVersion = "1.2.0";
contract.systemName = "ARES";
contract.vehicleScope = "single-uav";

contract.coordinateFrame = "NED";
contract.timeBase = "simulation-time";
contract.timeUnit = "seconds";

statusCodes = num2cell(uint8((0:3)'));

contract.staticFields = [
    "sample_time_s"
    "initial_position_ned_m"
    "waypoints_ned_m"
    "vehicle_radius_m"
    "avoidance_safety_distance_m"
    "maximum_sensor_range_m"
    "model_name"
    "git_commit"];

contract.dynamicFields = [
    "time_s"
    "position_ned_m"
    "desired_position_ned_m"
    "velocity_ned_mps"
    "command_roll_rad"
    "command_pitch_rad"
    "command_yaw_rate_radps"
    "command_thrust_n"
    "obstacle_avoidance_status"
];

statusLabels = {
    "free_direction"
    "no_free_direction"
    "free_direction_close_to_obstacle"
    "no_free_direction_close_to_obstacle"};

statusDescriptions = {
    "An obstacle-free direction was found."
    "No obstacle-free direction was found."
    "A free direction was found, but it is close to an obstacle."
    "No free direction was found, and the UAV is close to an obstacle."
    };

contract.enumerations.obstacleAvoidanceStatus = struct( ...
    "code", statusCodes, ...
    "label", statusLabels, ...
    "description", statusDescriptions);


contract.groundTruthFields = [
    "scenario_name"
    "fault_type"
    "fault_active"
    "fault_start_time_s"
    "fault_end_time_s"
    "expected_mission_outcome"
    ];

contract.createdBy = "ares.data.defaultDataContract";

end