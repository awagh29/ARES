classdef TestMapTemporalPlanner < matlab.unittest.TestCase
    methods (TestClassSetup)
        function locate(~)
            root=string(fileparts(fileparts(fileparts(mfilename('fullpath')))));
            addpath(fullfile(root,'src'));
        end
    end
    methods (Test)
        function openSpaceAndReset(test)
            o=ares.mapTemporal.defaultOptions(); o.GoalState=[8 0 -7 0 0 0];
            planner=ares.mapTemporal.TrajectoryPlanner(o);
            [v,q]=planner.step([0 0 -7],[0 0 0],zeros(0,3));
            test.verifyGreaterThan(v(1),0); test.verifyEqual(v(2:3),[0;0],'AbsTol',1e-9);
            test.verifyTrue(all(isfinite(q))); test.verifyLessThanOrEqual(norm(v),o.CruiseSpeedMps);
            planner.reset(); [again,r]=planner.step([0 0 -7],[0 0 0],zeros(0,3));
            test.verifyEqual(again,v); test.verifyEqual(r(1:11),q(1:11));
        end
        function blockedDirectionChoosesLateralSpace(test)
            o=ares.mapTemporal.defaultOptions(); o.GoalState=[8 0 -7 0 0 0];
            planner=ares.mapTemporal.TrajectoryPlanner(o);
            [y,z]=ndgrid(-2:.2:2,-2:.2:2); wall=[2*ones(numel(y),1),y(:),z(:)];
            [v,q]=planner.step([0 0 -7],[0 0 0],wall);
            test.verifyGreaterThan(q(4),0);
            test.verifyGreaterThan(abs(v(2)),abs(v(1)));
            test.verifyGreaterThanOrEqual(q(6),o.MinimumSurfaceClearanceM);
        end
        function stalledHistoryCommitsToEscape(test)
            o=ares.mapTemporal.defaultOptions(); o.GoalState=[8 0 -7 0 0 0];
            planner=ares.mapTemporal.TrajectoryPlanner(o);
            for k=1:30, [~,q]=planner.step([0 0 -7],[0 0 0],zeros(0,3)); end
            test.verifyEqual(q(1),1); test.verifyEqual(q(3),1);
            target=q(8:10);
            [~,r]=planner.step([.05 0 -7],[0 0 0],zeros(0,3));
            test.verifyEqual(r(8:10),target); test.verifyEqual(r(3),1);
        end
        function momentumAndBoundsRejectUnstoppableMotion(test)
            o=ares.mapTemporal.defaultOptions(); o.GoalState=[8 0 -7 0 0 0];
            planner=ares.mapTemporal.TrajectoryPlanner(o);
            [y,z]=ndgrid(-4:.2:4,-2:.2:2); wall=[1.7*ones(numel(y),1),y(:),z(:)];
            [v,q]=planner.step([0 0 -7],[2 0 0],wall);
            test.verifyEqual(v,zeros(3,1)); test.verifyEqual(q(11),1);
            o.PlannerVoxelM=1;
            test.verifyError(@()ares.mapTemporal.validateOptions(o),'ARES:TemporalPlannerOption');
        end
        function stopTailAccountsForDelayedBraking(test)
            o=ares.mapTemporal.defaultOptions(); o.GoalState=[20 0 -7 0 0 0];
            o.StartState(4)=1.5;
            planner=ares.mapTemporal.TrajectoryPlanner(o);
            [y,z]=ndgrid(-8:.2:8,-2:.2:2); wall=[6.2*ones(numel(y),1),y(:),z(:)];
            [v,q]=planner.step([0 0 -7],[1.5 0 0],wall);
            test.verifyLessThan(v(1),1.4);
            test.verifyGreaterThan(q(4),0);
        end
        function escapeTimeoutRestartsNormalProgressWindow(test)
            o=ares.mapTemporal.defaultOptions(); o.GoalState=[8 0 -7 0 0 0];
            planner=ares.mapTemporal.TrajectoryPlanner(o);
            for k=1:30, [~,q]=planner.step([0 0 -7],[0 0 0],zeros(0,3)); end
            test.verifyEqual(q(3),1);
            for k=1:80
                [~,q]=planner.step([0 0 -7],[0 0 0],zeros(0,3));
                if q(1)==0, break; end
            end
            test.verifyEqual(q(1),0); test.verifyEqual(q(3),1);
            for k=1:5, [~,q]=planner.step([0 0 -7],[0 0 0],zeros(0,3)); end
            test.verifyEqual(q(1),0); test.verifyEqual(q(3),1);
        end
        function bypassMotionIsNotStationaryStall(test)
            o=ares.mapTemporal.defaultOptions(); o.GoalState=[8 0 -7 0 0 0];
            planner=ares.mapTemporal.TrajectoryPlanner(o);
            for k=1:30
                [~,q]=planner.step([0 .1*k -7],[0 1 0],zeros(0,3));
            end
            test.verifyEqual(q(2),0); test.verifyEqual(q(3),0);
        end
    end
end
