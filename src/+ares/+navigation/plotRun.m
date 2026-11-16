function path = plotRun(result)
%PLOTRUN Export actual 3-D flight, clearance and outage evidence to one image.
data=load(fullfile(result.Directory,'telemetry.mat'),'telemetry','scans');
mission=load(result.PlanFile,'map','plan','options');
state=data.telemetry.signals.position_ned_m;
time=double(state.time_s(:)); p=double(state.values); scan=data.scans;
figureHandle=figure('Visible','off','Color','w','Position',[100 100 1400 760]);
cleanup=onCleanup(@()close(figureHandle)); %#ok<NASGU>
layout=tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(layout,1,[2 1]); hold(ax,'on');
faces=[1 2 3 4;5 6 7 8;1 2 6 5;2 3 7 6;3 4 8 7;4 1 5 8];
for k=1:size(mission.map.BoxMin,1)
    lo=mission.map.BoxMin(k,:); hi=mission.map.BoxMax(k,:);
    vertices=[lo;hi(1) lo(2) lo(3);hi(1) hi(2) lo(3);lo(1) hi(2) lo(3); ...
        lo(1) lo(2) hi(3);hi(1) lo(2) hi(3);hi;lo(1) hi(2) hi(3)];
    patch(ax,'Vertices',[vertices(:,2),vertices(:,1),-vertices(:,3)],'Faces',faces, ...
        'FaceColor',[.65 .65 .65],'FaceAlpha',.22,'EdgeColor',[.5 .5 .5],'HandleVisibility','off');
end
route=mission.plan.WaypointsNedM;
plot3(ax,route(:,2),route(:,1),-route(:,3),'k--','DisplayName','Planned route','LineWidth',1.1);
plot3(ax,p(:,2),p(:,1),-p(:,3),'Color',[0 .35 .65],'DisplayName','Actual flight','LineWidth',2);
unavailable=interp1(scan.time_s,double(~logical(scan.lidar_available)),time,'previous','extrap')>.5;
outage=p; outage(~unavailable,:)=NaN;
plot3(ax,outage(:,2),outage(:,1),-outage(:,3),'Color',[.85 .35 0], ...
    'DisplayName','Flight during outage','LineWidth',2.5);
scatter3(ax,p(1,2),p(1,1),-p(1,3),65,[0 .5 .2],'filled','DisplayName','Start');
goal=mission.options.GoalState(1:3);
scatter3(ax,goal(2),goal(1),-goal(3),85,[.6 0 .2],'filled','DisplayName','Goal');
xlabel(ax,'East (m)'); ylabel(ax,'North (m)'); zlabel(ax,'Altitude (m)');
title(ax,'Flight through the existing obstacle map'); grid(ax,'on'); axis(ax,'equal'); view(ax,40,28);
legend(ax,'Location','best');
clearance=Inf(numel(time),1);
for k=1:size(mission.map.BoxMin,1)
    center=(mission.map.BoxMin(k,:)+mission.map.BoxMax(k,:))/2;
    half=(mission.map.BoxMax(k,:)-mission.map.BoxMin(k,:))/2;
    q=abs(p-center)-half;
    clearance=min(clearance,vecnorm(max(q,0),2,2)+min(max(q,[],2),0)-mission.options.VehicleRadiusM);
end
ax=nexttile(layout,2);
plot(ax,time,clearance,'Color',[0 .35 .65],'LineWidth',1.4); hold(ax,'on');
yline(ax,mission.options.MinimumSurfaceClearanceM,'r--','Required clearance');
xlabel(ax,'Simulation time (s)'); ylabel(ax,'Surface clearance (m)');
title(ax,sprintf('Minimum measured flight clearance: %.2f m',min(clearance))); grid(ax,'on');
ax=nexttile(layout,4);
stairs(ax,scan.time_s,scan.lidar_available,'Color',[0 .35 .65],'LineWidth',1.3); hold(ax,'on');
stairs(ax,scan.time_s,scan.map_outage_correction_used,'Color',[.85 .35 0],'LineWidth',1.3);
ylim(ax,[-.1 1.1]); yticks(ax,[0 1]); xlabel(ax,'Simulation time (s)'); ylabel(ax,'Active (0 / 1)');
legend(ax,{'Measured LiDAR available','Map outage correction used'},'Location','best'); grid(ax,'on');
title(ax,sprintf('Observed outages: %d | All filters feasible: %d', ...
    result.ObservedOutageCount,result.Proof.AllFiltersFeasible));
mainTitle=title(layout,sprintf('%s | Goal reached at rest: %d | Arrival %.2f s | Path %.2f m', ...
    upper(result.LidarMode),result.Metrics.missionSuccess,result.ArrivalTimeS,result.Metrics.totalPathLengthM));
mainTitle.Color=[.12 .12 .12];
caption=subtitle(layout,'Source: executed Simulink plant telemetry; static supplied map; sampled validation');
caption.Color=[.3 .3 .3];
% Keep exports legible when the user's MATLAB desktop uses a dark theme.
set(findall(figureHandle,'Type','axes'),'Color','w','XColor','k','YColor','k','ZColor','k','GridColor',[.7 .7 .7]);
set(findall(figureHandle,'Type','text'),'Color',[.12 .12 .12]);
set(findall(figureHandle,'Type','legend'),'Color','w','TextColor','k','EdgeColor',[.5 .5 .5]);
path=fullfile(result.Directory,'flight_evidence.png'); exportgraphics(figureHandle,path,'Resolution',140);
end
