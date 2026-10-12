function contract = defaultDataContract()
% For one UAV ONLY!!
contract.schemaVersion = "2.0.0";
contract.systemName = "ARES";
contract.vehicleScope = "single-uav";

contract.coordinateFrame = "NED";
contract.timeBase = "simulation-time";
contract.timeUnit = "seconds";

statusCodes = num2cell(uint8((0:3)'));

contract.staticFields = [
    % Episode and experimental condition
    "episode_id"
    "trial_group_id"
    "monte_carlo_replicate_index"
    "scenario_type"
    "environment_id"
    "obstacle_layout_id"
    "random_seed"
    "model_name"

    % Mathematical spaces
    "state_space_definition"
    "state_vector_definition"
    "action_space_definition"
    "action_vector_definition"
    "goal_definition"

    % Observation spaces
    "nominal_observation_schema"
    "nominal_observation_dimension"
    "degraded_observation_schema"
    "reconstructable_channel_indices"
    "observation_channel_names"
    "observation_normalization_parameters"

    % System dynamics and observation model
    "dynamics_model_id"
    "dynamics_model_parameters"
    "observation_kernel_model_id"
    "observation_kernel_parameters"

    % Initial condition
    "initial_state_mean"
    "initial_state_covariance"
    "initial_condition_distribution"
    "initial_position_ned_m"
    "initial_velocity_ned_mps"
    "initial_attitude_rpy_rad"

    % Mission definition
    "waypoints_ned_m"
    "goal_region_definition"
    "goal_tolerance_m"
    "navigation_horizon_steps"
    "maximum_episode_duration_s"

    % Formal safety definition
    "unsafe_state_set_definition"
    "safety_risk_tolerance_delta"
    "reconstruction_confidence_level"

    % Environment
    "map_bounds_ned_m"
    "map_resolution_m"
    "unknown_space_policy"
    "vehicle_radius_m"
    "avoidance_safety_distance_m"
    "obstacle_ground_truth_reference"
    "scene_distribution_id"

    % LiDAR definition
    "maximum_sensor_range_m"
    "lidar_horizontal_fov_rad"
    "lidar_vertical_fov_rad"
    "lidar_angular_resolution_rad"
    "lidar_update_rate_hz"
    "lidar_mounting_pose"
    "lidar_measurement_noise_mean"
    "lidar_measurement_noise_covariance"

    % Failure configuration
    "failure_configuration_id"
    "failure_mask_definition"
    "failure_channels"
    "failure_start_distribution"
    "failure_duration_distribution"
    "lidar_dropout_schedule"
    "lidar_dropout_probability"
    "lidar_minimum_off_duration_s"
    "lidar_maximum_off_duration_s"

    % Reconstruction model
    "reconstruction_model_id"
    "reconstruction_model_version"
    "reconstruction_model_type"
    "reconstruction_history_length"
    "reconstruction_prior_model_id"
    "reconstruction_loss_definition"
    "observation_loss_weights"
    "control_relevant_loss_weights"

    % Bayesian model
    "bayesian_hypothesis_definition"
    "bayesian_prior_parameters"
    "bayesian_likelihood_model"
    "bayesian_likelihood_parameters"
    "posterior_confidence_threshold"

    % Fixed nominal policy
    "nominal_policy_id"
    "nominal_policy_version"
    "nominal_policy_model_hash"
    "nominal_policy_input_schema"
    "nominal_policy_output_schema"

    % Planner, MPC and safety-filter configuration
    "planner_configuration_id"
    "controller_configuration_id"
    "kinodynamic_astar_parameters"
    "mpc_prediction_horizon"
    "mpc_control_horizon"
    "mpc_cost_weights"
    "mpc_constraint_parameters"
    "safety_filter_type"
    "barrier_function_parameters"
    "barrier_uncertainty_margin"

    % Timing
    "simulation_sample_time_s"
    "sensor_sample_time_s"
    "reconstruction_sample_time_s"
    "controller_sample_time_s"
];

contract.requiredLoggedSignals = [
    "position_ned_m"
    "desired_position_ned_m"
    "velocity_ned_mps"
    "command_roll_rad"
    "command_pitch_rad"
    "command_yaw_rate_radps"
    "command_thrust_n"
    "lidar_enabled"
    "obstacle_avoidance_status"
    "nominal_lidar_points_xyz_m"
    ];

contract.dynamicFields = [
    % Synchronization
    "time_s"
    "sample_index"
    "sensor_measurement_timestamp_s"
    "reconstruction_timestamp_s"
    "control_timestamp_s"

    % Sensor condition
    "lidar_commanded_on"
    "lidar_data_received"
    "lidar_data_valid"
    "lidar_failure_active"
    "lidar_failure_configuration"
    "lidar_dropout_event"
    "latest_lidar_scan_age_s"
    "time_since_last_valid_lidar_s"

    % Complete nominal observation o_t
    "nominal_observation"
    "nominal_lidar_ranges_m"
    "nominal_lidar_valid_return_mask"

    % Degraded observation tilde{o}_t
    "degraded_observation"
    "degraded_lidar_ranges_m"
    "observation_availability_mask"
    "missing_observation_mask"
    "available_observation_dimension"
    "missing_observation_dimension"

    % Ground-truth missing component — training only
    "missing_observation_ground_truth"

    % Reconstruction Gaussian prior
    "reconstruction_prior_mean"
    "reconstruction_prior_covariance"

    % Bayesian reconstruction calculation
    "reconstruction_measurement_likelihood"
    "reconstruction_log_likelihood"
    "reconstruction_evidence"
    "bayesian_hypothesis_priors"
    "bayesian_hypothesis_likelihoods"
    "bayesian_hypothesis_posteriors"
    "maximum_a_posteriori_hypothesis"
    "posterior_entropy"
    "posterior_confidence"

    % Observation posterior
    "reconstruction_posterior_mean"
    "reconstruction_posterior_covariance"
    "reconstruction_interval_lower"
    "reconstruction_interval_upper"
    "posterior_predictive_samples"

    % Reconstructed observation hat{o}_t
    "reconstructed_missing_observation"
    "reconstructed_observation"
    "reconstruction_valid"
    "reconstruction_computation_time_s"

    % Reconstruction quality
    "reconstruction_error"
    "reconstruction_rmse"
    "reconstruction_negative_log_likelihood"
    "reconstruction_coverage_indicator"
    "reconstruction_calibration_error"

    % State-estimation belief
    "state_prior_mean"
    "state_prior_covariance"
    "state_posterior_mean"
    "state_posterior_covariance"
    "position_estimated_ned_m"
    "velocity_estimated_ned_mps"
    "attitude_estimated_rpy_rad"
    "state_estimator_valid"

    % Simulator state x_t — evaluation only
    "state_true"
    "position_true_ned_m"
    "velocity_true_ned_mps"
    "attitude_true_rpy_rad"

    % Goal supplied to the policy
    "active_waypoint_index"
    "goal_input"
    "relative_goal_position_body_m"
    "distance_to_goal_m"

    % Nominal-policy comparisons
    "nominal_action_full_observation"
    "nominal_action_degraded_observation"
    "nominal_action_reconstructed_observation"
    "action_reconstruction_error"

    % Optional planner and MPC outputs
    "trajectory_valid"
    "replan_requested"
    "replan_executed"
    "replan_reason"
    "planner_status"
    "planned_trajectory_reference"
    "mpc_feasible"
    "mpc_nominal_command"

    % Safety-filter separation
    "pre_filter_command"
    "post_filter_command"
    "safety_filter_active"
    "safety_filter_feasible"
    "minimum_barrier_value"
    "barrier_constraint_slack"
    "safety_filter_correction_magnitude"

    % Executed action a_t
    "executed_action"
    "command_roll_rad"
    "command_pitch_rad"
    "command_yaw_rate_radps"
    "command_thrust_n"
    "action_source"

    % Formal safety and success indicators
    "in_goal_region"
    "in_goal_region_at_horizon"
    "first_goal_entry_time_s"
    "in_unsafe_state"
    "entered_unsafe_state_so_far"
    "collision_detected"
    "near_collision_detected"
    "minimum_obstacle_clearance_m"

    % Probabilistic safety quantities
    "predicted_unsafe_probability"
    "predicted_goal_probability"
    "estimated_chance_constraint_value"
    "risk_budget_remaining"
    "chance_constraint_satisfied"

    % Episode state
    "episode_done"
    "termination_reason"
];

contract.episodeSummaryFields = [
    "episode_id"
    "scenario_type"
    "reconstruction_model_id"

    "in_goal_region_at_horizon"
    "goal_reached_at_any_time"
    "mission_success_event"
    "unsafe_state_visited"
    "collision_occurred"

    "completion_time_s"
    "total_path_length_m"
    "minimum_obstacle_clearance_m"

    "mean_reconstruction_rmse"
    "mean_reconstruction_nll"
    "posterior_interval_coverage"
    "mean_action_reconstruction_error"

    "number_of_replans"
    "number_of_safety_interventions"
    "total_lidar_outage_time_s"
    "maximum_continuous_lidar_outage_s"
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