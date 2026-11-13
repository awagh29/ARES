function [schedule, intervals, faultConfiguration] = createIntermittentSchedule(simulationStopTime)
%CREATEINTERMITTENTSCHEDULE Preserve the original seed-513 outage experiment.
% Same draws, ordering, ZOH edges and parameters as the local generator in
% scripts/captureIntermittentLidarDisabledSingleUav.m. Uses a private RNG.
arguments
    simulationStopTime (1,1) double {mustBePositive,mustBeFinite} = 60
end
faultConfiguration.randomSeed = 513;
faultConfiguration.numberOfOutages = 3;
faultConfiguration.initialHealthyTime_s = 3;
faultConfiguration.finalHealthyTime_s = 4;
faultConfiguration.minimumOutageDuration_s = 1.5;
faultConfiguration.maximumOutageDuration_s = 3.0;
faultConfiguration.minimumRecoveryDuration_s = 2.0;
faultConfiguration.faultWindowEndTime_s = 21;
faultWindowEndTime = min(simulationStopTime,faultConfiguration.faultWindowEndTime_s);
outageCount = faultConfiguration.numberOfOutages;
minimumRequiredTime = faultConfiguration.initialHealthyTime_s ...
    + outageCount*faultConfiguration.minimumOutageDuration_s ...
    + max(outageCount-1,0)*faultConfiguration.minimumRecoveryDuration_s ...
    + faultConfiguration.finalHealthyTime_s;
assert(faultWindowEndTime>=minimumRequiredTime,"ARES:InsufficientFaultScheduleTime", ...
    "The %.2f-second fault window is too short; at least %.2f seconds are required.", ...
    faultWindowEndTime,minimumRequiredTime);
randomStream = RandStream("mt19937ar","Seed",faultConfiguration.randomSeed);
intervals = zeros(outageCount,2);
earliestStartTime = faultConfiguration.initialHealthyTime_s;
for outageIndex = 1:outageCount
    remainingOutages = outageCount-outageIndex;
    remainingMinimumTime = faultConfiguration.finalHealthyTime_s ...
        + remainingOutages*faultConfiguration.minimumOutageDuration_s ...
        + remainingOutages*faultConfiguration.minimumRecoveryDuration_s;
    latestStartTime = faultWindowEndTime ...
        - faultConfiguration.minimumOutageDuration_s-remainingMinimumTime;
    assert(latestStartTime>=earliestStartTime,"ARES:InvalidFaultScheduleWindow", ...
        "No valid start-time window remains for outage %d.",outageIndex);
    startTime = earliestStartTime ...
        + rand(randomStream)*(latestStartTime-earliestStartTime);
    maximumAllowedDuration = min(faultConfiguration.maximumOutageDuration_s, ...
        faultWindowEndTime-startTime-remainingMinimumTime);
    assert(maximumAllowedDuration>=faultConfiguration.minimumOutageDuration_s, ...
        "ARES:InvalidFaultDurationWindow","No valid outage duration remains.");
    outageDuration = faultConfiguration.minimumOutageDuration_s ...
        + rand(randomStream)*(maximumAllowedDuration-faultConfiguration.minimumOutageDuration_s);
    endTime = startTime+outageDuration;
    intervals(outageIndex,:) = [startTime,endTime];
    earliestStartTime = endTime+faultConfiguration.minimumRecoveryDuration_s;
end
assert(intervals(end,2)+faultConfiguration.finalHealthyTime_s<=faultWindowEndTime+1e-9, ...
    "ARES:MissingFinalHealthyWindow","The final healthy interval was not preserved.");
transitionTimes = [0;reshape(intervals.',[],1);simulationStopTime];
availabilityValues = [1;repmat([0;1],outageCount,1);1];
assert(all(diff(transitionTimes)>0),"ARES:NonIncreasingFaultSchedule", ...
    "LiDAR schedule timestamps must increase strictly.");
schedule = timeseries(availabilityValues,transitionTimes);
schedule = setinterpmethod(schedule,"zoh");
schedule.Name = "lidar_enabled";
end
