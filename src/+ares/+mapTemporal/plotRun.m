function plotRun(telemetry,scans,map,options,folder)
%PLOTRUN Save the actual plant trajectory, fault timing and terminal errors.
visibility='off'; if options.ShowSimulation, visibility='on'; end
figureHandle=figure('Name','Map-temporal CNN single UAV','Visible',visibility,'Color','w');
layout=tiledlayout(figureHandle,2,2,'TileSpacing','compact');
ax=nexttile(layout,[2 1]); hold(ax,'on');
for k=1:size(map.BoxMin,1)
    lo=map.BoxMin(k,:); hi=map.BoxMax(k,:);
    vertices=[lo;hi(1) lo(2:3);hi(1:2) lo(3);lo(1) hi(2) lo(3); ...
        lo(1:2) hi(3);hi(1) lo(2) hi(3);hi;lo(1) hi(2:3)];
    vertices(:,3)=-vertices(:,3);
    patch(ax,'Vertices',vertices,'Faces',[1 2 3 4;5 6 7 8;1 2 6 5;2 3 7 6;3 4 8 7;4 1 5 8], ...
        'FaceColor',[.6 .65 .72],'FaceAlpha',.25,'EdgeColor',[.5 .5 .5],'HandleVisibility','off');
end
p=double(telemetry.signals.position_ned_m.values);
plot3(ax,p(:,1),p(:,2),-p(:,3),'b-','LineWidth',1.5,'DisplayName','Actual flight');
s=options.StartState; g=options.GoalState;
plot3(ax,s(1),s(2),-s(3),'go','MarkerFaceColor','g','DisplayName','Start');
plot3(ax,g(1),g(2),-g(3),'rx','LineWidth',2,'MarkerSize',10,'DisplayName','Goal');
grid(ax,'on'); axis(ax,'equal'); view(ax,3); xlabel(ax,'North (m)'); ylabel(ax,'East (m)'); zlabel(ax,'Altitude (m)');
legend(ax,'Location','best'); title(ax,'Original VFH/PID with learned LiDAR reconstruction');
ax=nexttile(layout); stairs(ax,scans.time_s,scans.lidar_available,'LineWidth',1.3); hold(ax,'on');
stairs(ax,scans.time_s,scans.reconstruction_used,'--','LineWidth',1.3);
ylim(ax,[-.1 1.1]); grid(ax,'on'); legend(ax,'Measured available','CNN used'); xlabel(ax,'Time (s)');
ax=nexttile(layout); plot(ax,scans.time_s,scans.goal_position_error_m,'LineWidth',1.3); hold(ax,'on');
plot(ax,scans.time_s,scans.goal_velocity_error_mps,'LineWidth',1.3);
grid(ax,'on'); legend(ax,'Position error (m)','Velocity error (m/s)'); xlabel(ax,'Time (s)');
% Keep saved figures readable even when MATLAB's desktop uses a dark theme.
for axesHandle=findall(figureHandle,'Type','axes')'
    set(axesHandle,'Color','w','XColor','k','YColor','k','ZColor','k');
    set([axesHandle.Title,axesHandle.XLabel,axesHandle.YLabel,axesHandle.ZLabel],'Color','k');
end
set(findall(figureHandle,'Type','legend'),'Color','w','TextColor','k');
exportgraphics(figureHandle,fullfile(folder,'mission.png'),'Resolution',160);
if ~options.ShowSimulation, close(figureHandle); end
end
