function [schedule,intervals,fault] = denseSaitsLidarScenario(stopTime,lidarMode)
%DENSESAITSLIDARSCENARIO Explicit measured-data availability for SAITS tests.
arguments
    stopTime (1,1) double {mustBeFinite,mustBePositive}
    lidarMode (1,1) string {mustBeMember(lidarMode,["intermittent","disabled"])} = "intermittent"
end
if lidarMode=="intermittent"
    [schedule,intervals,fault]=ares.inference.createIntermittentSchedule(stopTime);
else
    % No healthy first sample and no recovery at the end. This is not an
    % intermittent fault with a long duration or a preloaded healthy buffer.
    schedule=timeseries(zeros(2,1),[0;stopTime]);
    schedule=setinterpmethod(schedule,"zoh");
    schedule.Name="lidar_enabled";
    intervals=[0 stopTime];
    fault=struct("randomSeed",[],"numberOfOutages",1, ...
        "initialHealthyTime_s",0,"finalHealthyTime_s",0, ...
        "minimumOutageDuration_s",stopTime,"maximumOutageDuration_s",stopTime, ...
        "minimumRecoveryDuration_s",0,"faultWindowEndTime_s",stopTime);
end
fault.lidarMode=lidarMode;
end
