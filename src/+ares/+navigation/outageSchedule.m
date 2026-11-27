function [schedule,intervals] = outageSchedule(options)
%OUTAGESCHEDULE Quantized, reproducible outages; blackout needs no warm start.
o=ares.navigation.validateOptions(options); T=o.StopTimeS; tick=.1;
switch o.LidarMode
    case "blackout"
        intervals=[0 T];
    case "healthy"
        intervals=zeros(0,2);
    case "custom"
        intervals=o.OutageIntervalsS;
    otherwise
        n=o.NumberOfOutages; intervals=zeros(n,2);
        if n>0
            first=ceil(o.InitialHealthyTimeS/tick);
            tail=ceil(o.FinalHealthyTimeS/tick);
            gap=max(1,ceil(o.MinimumRecoveryTimeS/tick));
            last=floor(min(T,o.OutageWindowEndS)/tick);
            lo=ceil(o.OutageDurationS(1)/tick); hi=floor(o.OutageDurationS(2)/tick);
            assert(hi>=lo,'ARES:MapDurationResolution','No duration fits the 0.1 s scan grid.');
            budget=last-first-tail-(n-1)*gap;
            assert(budget>=n*lo,'ARES:MapScheduleTooShort', ...
                'Not enough time for %d outages. Reduce count/durations or increase OutageWindowEndS and StopTimeS.',n);
            stream=RandStream('mt19937ar','Seed',o.Seed);
            durations=lo*ones(n,1);
            for k=1:n
                extra=min(hi-lo,budget-sum(durations));
                durations(k)=durations(k)+randi(stream,extra+1)-1;
            end
            spare=budget-sum(durations);
            weights=rand(stream,n+1,1); gaps=floor(spare*weights/sum(weights));
            gaps(end)=gaps(end)+spare-sum(gaps);
            cursor=first+gaps(1);
            for k=1:n
                intervals(k,:)=[cursor,cursor+durations(k)]*tick;
                cursor=cursor+durations(k)+gap+gaps(k+1);
            end
        end
end
assert(all(intervals(:,1)>=0 & intervals(:,2)>intervals(:,1) & intervals(:,2)<=T), ...
    'ARES:InvalidMapIntervals','Outage intervals must satisfy 0 <= start < end <= StopTimeS.');
assert(isempty(intervals) || all(intervals(2:end,1)>intervals(1:end-1,2)), ...
    'ARES:OverlappingMapIntervals','Outages must be sorted, disjoint, and separated by a healthy interval.');
assert(all(abs(intervals(:)/tick-round(intervals(:)/tick))<1e-7), ...
    'ARES:MapIntervalResolution','Outage edges must be multiples of 0.1 s.');
times=unique([0;intervals(:);T]); enabled=ones(size(times));
for k=1:size(intervals,1)
    enabled(times>=intervals(k,1) & times<intervals(k,2))=0;
end
if ~isempty(intervals) && intervals(end,2)==T, enabled(end)=0; end
schedule=timeseries(enabled,times); schedule=setinterpmethod(schedule,'zoh');
schedule.Name='lidar_enabled';
end
