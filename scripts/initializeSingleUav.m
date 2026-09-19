cfg = ares.config.defaultSingleUav();

UAVSampleTime = cfg.simulation.sampleTime;
DroneMass = cfg.vehicle.massKg;
Gravity = cfg.vehicle.gravityMps2;
VehicleRadius = cfg.vehicle.radiusM;

AvoidanceSafetyDistance = cfg.avoidance.planningSafetyDistanceM;

InitialPosition = cfg.mission.initialPositionNed;
InitialOrientation = [0 0 0];
Waypoints = cfg.mission.waypointsNed;

AzimuthResolution = 0.5;
ElevationResolution = 2;

MaxRange = 7;
RangeAccuracy = 3;

AzimuthLimits = [-179 179];
ElevationLimits = [-15 15];

Scenario = uavScenario( ...
"UpdateRate", 100, ...
"ReferenceLocation", [0 0 0]);

platUAV = uavPlatform(...
    UAV=Scenario, ...
    ReferenceFrame="NED", ...
    InitialPosition=InitialPosition, ...
    InitialOrientation=eul2quat(InitialOrientation));

updateMesh( ...
    platUAV, ...
    "quadrotor", ...
    {1.2}, ...
    [0 0 1], ...
    eul2tform([0 0 pi]));

LidarModel = uavLidarPointCloudGenerator( ...
    UpdateRate=10, ...
    MaxRange=MaxRange, ...
    RangeAccuracy=RangeAccuracy, ...
    AzimuthResolution=AzimuthResolution, ...
    ElevationResolution=ElevationResolution, ...
    AzimuthLimits=AzimuthLimits, ...
    ElevationLimits=ElevationLimits, ...
    HasOrganizedOutput=true);

uavSensor( ...
    "Lidar", ...
    platUAV, ...
    LidarModel, ...
    MountingLocation=[0 0 -0.4], ...
    MountingAngles=[0 0 180]);

ObstaclePositions = [
    10  0
    20 10
    10 20
    ];

ObstacleHeight = 15;
ObstacleWidth = 3;

for i = 1:size(ObstaclePositions, 1)

    north = ObstaclePositions(i, 1);
    east = ObstaclePositions(i, 2);

    vertices = [
        north - ObstacleWidth/2, east - ObstacleWidth/2
        north + ObstacleWidth/2, east - ObstacleWidth/2
        north + ObstacleWidth/2, east + ObstacleWidth/2
        north - ObstacleWidth/2, east + ObstacleWidth/2
        ];

    addMesh( ...
        Scenario, ...
        "polygon", ...
        {vertices, [0 ObstacleHeight]}, ...
        0.651 * ones(1,3));
end


Px = 6;
Py = 6;
Pz = 6.5;

Dx = 1.5;
Dy = 1.5;
Dz = 2.5;

Ix = 0;
Iy = 0;
Iz = 0;

Nx = 10;
Ny = 10;
Nz = 14.4947065605712;

disp("ARES Single UAV Intialized!")