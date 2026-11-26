function [direction, desiredYaw, status] = aresVfhStep(doReset, position, orientation, points, target) %#codegen
%ARESVFHSTEP The baseline model's "Obstacle Avoidance" block (UAV Toolbox 3DVFH+), callable as a MEX.
% Same object and the same settings as models/ares_single_uav_top.slx:
%   DistanceLimits [0.2 MaxRange=7], FOV [-179 179] x [-15 15], sensor at [0 0 -0.4] rolled 180 deg,
%   VehicleRadius 0.1, SafetyDistance 1.5, HistRes 6, WindowSize 1, HistogramThreshold 1,
%   MaxAge 3, weights target/current/previous = 5/2/2.
% buildVfhMex.m compiles it and checks the MEX against this interpreted version.
persistent avoider
if isempty(avoider)
    avoider = uav.sluav.internal.system.ObstacleAvoidance( ...
        'DistanceLimits', [.2 7], 'HorizontalSensorFOV', [-179 179], 'VerticalSensorFOV', [-15 15], ...
        'SensorLocation', [0 0 -.4], 'SensorOrientation', [180 0 0], ...
        'VehicleRadius', .1, 'SafetyDistance', 1.5, ...
        'HistRes', '6', 'WindowSize', 1, 'HistogramThreshold', 1, 'MaxAge', 3, ...
        'TargetDirectionWeight', 5, 'CurrentDirectionWeight', 2, 'PreviousDirectionWeight', 2);
end
if doReset
    reset(avoider);
end
[direction, desiredYaw, status] = avoider(position, orientation, points, target);
end
