function cfg = defaultSingleUav()
%Default Single UAV Central configuration for our first drone

cfg.identity.id = "uav_001";

cfg.frames.world = "world_ned";
cfg.frames.body = "uav_001/base_link";
cfg.frames.lidar = "uav_001/lidar_link";

cfg.simulation.sampleTime = 0.001;
cfg.simulation.stopTime = 60;

cfg.vehicle.massKg = 0.1;
cfg.vehicle.gravityMps2 = 9.81;
cfg.vehicle.radiusM = 0.1;

cfg.mission.initialPositionNed = [0 0 -7];

cfg.mission.waypointsNed = [
    0   0  -7
    0  20  -7
    20  20  -7
    20   0  -7
    ];

cfg.safety.maxHorizontalSpeedMps = 3.0;
cfg.safety.minimumClearanceM = 1.5;
cfg.safety.brakingDecelerationMps2 = 2.0;
cfg.safety.sensorLatencySeconds = 0.2;
cfg.safety.additionalMarginM = 0.5;
cfg.safety.minimumObstacleDistanceM = 1.0;

cfg.acceptance.finalWaypointToleranceM = 1.0;
cfg.acceptance.minimumAltitudeM = 5.0;
cfg.acceptance.maximumAltitudeM = 9.0;

cfg.avoidance.planningSafetyDistanceM = 1.5;

end
