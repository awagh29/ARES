function plotPlannerComparison(control,result,path)
%PLOTPLANNERCOMPARISON Actual plant traces and native-rate clearance, not rollouts.
before=readtable(fullfile(control.Directory,'scans.csv')); after=readtable(fullfile(result.Directory,'scans.csv'));
mission=load(result.PlanFile,'map','options'); map=mission.map; o=mission.options;
f=figure('Visible','off','Color','w','Position',[100 100 1150 780]); clean=onCleanup(@()close(f)); %#ok<NASGU>
tiledlayout(f,2,2,'TileSpacing','compact');
ax=nexttile; plot(ax,before.position_n,before.position_e,'--',after.position_n,after.position_e,'LineWidth',1.3); hold(ax,'on');
for k=1:size(map.BoxMin,1)
    lo=map.BoxMin(k,:); hi=map.BoxMax(k,:);
    rectangle(ax,'Position',[lo(1:2),hi(1:2)-lo(1:2)],'EdgeColor',[.4 .4 .4]);
end
plot(ax,o.GoalState(1),o.GoalState(2),'kx','LineWidth',1.5,'HandleVisibility','off');
axis(ax,'equal'); grid(ax,'on'); xlabel(ax,'North (m)'); ylabel(ax,'East (m)'); title(ax,'Actual trajectory');
legend(ax,'Governor only','Trajectory planner','Location','best');
ax=nexttile; hold(ax,'on'); cases={control,result}; styles={'--','-'};
for arm=1:2
    d=load(fullfile(cases{arm}.Directory,'telemetry.mat'),'telemetry'); s=d.telemetry.signals.position_ned_m;
    p=double(s.values); clearance=Inf(size(p,1),1);
    for k=1:size(map.BoxMin,1)
        center=(map.BoxMin(k,:)+map.BoxMax(k,:))/2; half=(map.BoxMax(k,:)-map.BoxMin(k,:))/2;
        q=abs(p-center)-half;
        clearance=min(clearance,vecnorm(max(q,0),2,2)+min(max(q,[],2),0)-o.VehicleRadiusM);
    end
    plot(ax,s.time_s,clearance,styles{arm},'LineWidth',1.3);
end
yline(ax,o.MinimumSurfaceClearanceM,':','Required','Color',[.3 .3 .3]);
grid(ax,'on'); xlabel(ax,'Time (s)'); ylabel(ax,'Minimum surface clearance (m)');
ax=nexttile; plot(ax,before.time_s,before.goal_position_error_m,'--',after.time_s,after.goal_position_error_m,'LineWidth',1.3);
yline(ax,o.GoalPositionToleranceM,':','Tolerance','Color',[.3 .3 .3]); grid(ax,'on'); xlabel(ax,'Time (s)'); ylabel(ax,'Goal position error (m)');
ax=nexttile; plot(ax,before.time_s,before.goal_velocity_error_mps,'--',after.time_s,after.goal_velocity_error_mps,'LineWidth',1.3);
yline(ax,o.GoalVelocityToleranceMps,':','Tolerance','Color',[.3 .3 .3]); grid(ax,'on'); xlabel(ax,'Time (s)'); ylabel(ax,'Goal velocity error (m/s)');
sgtitle(f,sprintf('%s LiDAR: trajectory planner and stall recovery | frozen reconstruction',result.LidarMode),'Color','k');
for ax=findall(f,'Type','axes')'
    set(ax,'Color','w','XColor','k','YColor','k'); set([ax.Title,ax.XLabel,ax.YLabel],'Color','k');
end
set(findall(f,'Type','legend'),'Color','w','TextColor','k'); exportgraphics(f,path,'Resolution',140);
end
