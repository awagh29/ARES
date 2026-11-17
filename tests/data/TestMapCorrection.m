classdef TestMapCorrection < matlab.unittest.TestCase
    properties
        Root
        Map
        Options
    end
    methods (TestClassSetup)
        function setupPaths(testCase)
            testCase.Root=string(fileparts(fileparts(fileparts(mfilename('fullpath')))));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(testCase.Root,'src')));
            testCase.Map=ares.navigation.sceneMap([10 0;15 15;10 20;5 15;22 18;15 25;10 20],3,15);
            testCase.Options=ares.navigation.defaultOptions(); testCase.Options.ShowSimulation=false;
        end
    end
    methods (Test)
        function existingMapConvertsEnuToNed(testCase)
            testCase.verifyEqual(size(testCase.Map.BoxMin),[6 3]);
            testCase.verifyEqual(testCase.Map.BoxMin(1,:),[-1.5 8.5 -15]);
            testCase.verifyEqual(testCase.Map.BoxMax(1,:),[1.5 11.5 0]);
        end
        function blackoutStartsMissingAndNeverRecovers(testCase)
            o=testCase.Options; o.LidarMode="blackout";
            [s,intervals]=ares.navigation.outageSchedule(o);
            testCase.verifyEqual(intervals,[0 60]);
            testCase.verifyEqual(double(s.Time),[0;60]); testCase.verifyEqual(double(s.Data),[0;0]);
        end
        function outagesAreExactReproducibleAndDoNotChangeRng(testCase)
            o=testCase.Options; o.NumberOfOutages=4;
            before=rng; [a,intervals]=ares.navigation.outageSchedule(o); after=rng;
            [b,again]=ares.navigation.outageSchedule(o);
            testCase.verifyEqual(before,after); testCase.verifyEqual(intervals,again);
            testCase.verifyEqual(a.Data,b.Data); testCase.verifySize(intervals,[4 2]);
            testCase.verifyGreaterThanOrEqual(diff(intervals,1,2),o.OutageDurationS(1)*ones(4,1));
            testCase.verifyLessThanOrEqual(diff(intervals,1,2),o.OutageDurationS(2)*ones(4,1));
            testCase.verifyEqual(intervals/.1,round(intervals/.1),'AbsTol',1e-10);
        end
        function zeroOutagesAndHealthyModeRemainMeasured(testCase)
            o=testCase.Options; o.NumberOfOutages=0;
            [s,intervals]=ares.navigation.outageSchedule(o);
            testCase.verifyEmpty(intervals); testCase.verifyEqual(s.Data,[1;1]);
            o.LidarMode="healthy"; o.NumberOfOutages=9;
            [~,intervals]=ares.navigation.outageSchedule(o); testCase.verifyEmpty(intervals);
        end
        function customIntervalsAndImpossibleScheduleAreValidated(testCase)
            o=testCase.Options; o.LidarMode="custom"; o.OutageIntervalsS=[0 2;5 9];
            [s,intervals]=ares.navigation.outageSchedule(o);
            testCase.verifyEqual(intervals,o.OutageIntervalsS);
            testCase.verifyEqual(interp1(s.Time,s.Data,[0 1 2 5 9],'previous'),[0 0 1 0 1]);
            o.OutageIntervalsS=[0 4;3 5];
            testCase.verifyError(@()ares.navigation.outageSchedule(o),'ARES:OverlappingMapIntervals');
            o=testCase.Options; o.NumberOfOutages=100;
            testCase.verifyError(@()ares.navigation.outageSchedule(o),'ARES:MapScheduleTooShort');
        end
        function raycastUsesActualSurfacesAndSensorMount(testCase)
            map=testCase.Map;
            map.SensorDirections=[0 -1 0;1 0 0;0 0 -1];
            [~,ranges,hits]=ares.navigation.virtualLidar(map,[0 3 -7],[0 0 0]);
            testCase.verifyEqual(ranges,[5.5;7;7],'AbsTol',1e-10);
            testCase.verifyEqual(hits,[true;false;false]);
        end
        function entireSegmentIsCheckedForDiagonalCollision(testCase)
            map=ares.navigation.sceneMap([0 0],2,4); bounds=[-5 5;-5 5;-8 -1];
            testCase.verifyFalse(ares.navigation.segmentFree([-3 -3 -2],[3 3 -2],map,.5,bounds));
            testCase.verifyTrue(ares.navigation.segmentFree([-3 -3 -6],[3 3 -6],map,.5,bounds));
        end
        function plannerUsesThirdDimensionWhenWallBlocksHorizontalRoute(testCase)
            map=ares.navigation.sceneMap([0 0],2,4);
            map.BoxMin=[-1 -10 -4]; map.BoxMax=[1 10 0];
            o=testCase.Options; o.StartState=[-4 0 -2 0 0 0]; o.GoalState=[4 0 -2 0 0 0];
            o.WorldBoundsNedM=[-5 5;-2 2;-8 -1];
            o.MinimumSurfaceClearanceM=.2; o.GeometryUncertaintyM=0; o.TrackingMarginM=0; o.PlanningExtraMarginM=.2;
            plan=ares.navigation.planPath(map,o);
            testCase.verifyLessThan(min(plan.WaypointsNedM(:,3)),-4.5);
            for k=2:size(plan.WaypointsNedM,1)
                testCase.verifyTrue(ares.navigation.segmentFree(plan.WaypointsNedM(k-1,:), ...
                    plan.WaypointsNedM(k,:),map,plan.PlanningPaddingM,o.WorldBoundsNedM));
            end
        end
        function blockedEndpointFailsBeforeFlight(testCase)
            o=testCase.Options; o.StartState=[0 10 -7 0 0 0];
            testCase.verifyError(@()ares.navigation.planPath(testCase.Map,o),'ARES:MapEndpointBlocked');
        end
        function qpCorrectsApproachAndHonorsActuationBounds(testCase)
            o=testCase.Options; previous=[0 -.5 0];
            [u,info]=ares.navigation.correctMotion([0 5.8 -7],[0 .8 0],[0 1.5 0],previous,testCase.Map,o);
            testCase.verifyTrue(info.Feasible); testCase.verifyLessThan(u(2),-.4);
            testCase.verifyLessThanOrEqual(max(abs(u-previous)),o.JerkLimitMps3*o.ControlSampleTimeS+1e-8);
            testCase.verifyLessThanOrEqual(info.MaximumViolation,1e-6);
        end
        function unsafeStateIsNotLabeledAsFeasible(testCase)
            [u,info]=ares.navigation.correctMotion([0 10 -7],[0 0 0],[0 0 0],[0 0 0], ...
                testCase.Map,testCase.Options);
            testCase.verifyFalse(info.Feasible); testCase.verifyTrue(all(isfinite(u)));
        end
        function projectionSolvesIntersectionAndRejectsEmptySet(testCase)
            [u,ok]=ares.navigation.projectControl([2 2 2],[1 1 0;0 1 1],[1;1],-3*ones(3,1),3*ones(3,1));
            testCase.verifyTrue(ok); testCase.verifyEqual(u,[1;0;1],'AbsTol',1e-8);
            [~,ok]=ares.navigation.projectControl([0 0 0],[1 0 0;-1 0 0],[-1;-1],-3*ones(3,1),3*ones(3,1));
            testCase.verifyFalse(ok);
        end
        function measuredSurfaceAtVehicleCenterIsRejected(testCase)
            p=[0 0 -7];
            [~,info]=ares.navigation.correctMotion(p,[0 0 0],[0 0 0],[0 0 0], ...
                testCase.Map,testCase.Options,p);
            testCase.verifyFalse(info.Feasible); testCase.verifyLessThan(info.MinimumH,0);
        end
        function blackoutIgnoresHiddenScanAndResetIsReproducible(testCase)
            options=testCase.Options; options.GoalState=[2 0 -7 0 0 0];
            map=testCase.Map; plan=ares.navigation.planPath(map,options);
            file=string(tempname(fullfile(testCase.Root,'tmp')))+".mat";
            save(file,'map','options','plan');
            a=ares.navigation.MapCorrectionController('PlanFile',char(file));
            b=ares.navigation.MapCorrectionController('PlanFile',char(file));
            cleanup=onCleanup(@()releaseBoth(a,b)); %#ok<NASGU>
            p=[0 0 -7]; v=[0 0 0]; e=[0 0 0];
            [ca,ua,pa,sa]=a(NaN(10740,3,'single'),p,v,e,false,0);
            [cb,ub,pb,sb]=b(single(.01)*ones(10740,3,'single'),p,v,e,false,0);
            indices=setdiff(1:43,28);
            testCase.verifyEqual(ca,cb); testCase.verifyEqual(ua,ub);
            testCase.verifyEqual(pa(indices),pb(indices)); testCase.verifyEqual(sa,sb);
            testCase.verifyEqual(pa(20:22),[0;1;1]);
            [~,again,packet]=a([],p+1,v,e,true,0); % duplicate tick must be cached
            testCase.verifyEqual(again,ua); testCase.verifyEqual(packet,pa);
            reset(a);
            [~,again,packet]=a([],p,v,e,false,0);
            testCase.verifyEqual(again,ua); testCase.verifyEqual(packet(indices),pa(indices));
        end
        function heldScanKeepsItsOriginalWorldCoordinates(testCase)
            options=testCase.Options; options.GoalState=[2 0 -7 0 0 0];
            map=testCase.Map; plan=ares.navigation.planPath(map,options);
            file=string(tempname(fullfile(testCase.Root,'tmp')))+".mat";
            save(file,'map','options','plan');
            a=ares.navigation.MapCorrectionController('PlanFile',char(file));
            b=ares.navigation.MapCorrectionController('PlanFile',char(file));
            cleanup=onCleanup(@()releaseBoth(a,b)); %#ok<NASGU>
            p=[0 0 -7]; v=[0 0 0]; e=[0 0 0]; cloud=ares.navigation.virtualLidar(map,p,e);
            a(cloud,p,v,e,true,0); b(cloud,p,v,e,true,0);
            [~,ua,pa]=a(cloud,p+[.01 0 0],v,e,true,.05);
            [~,ub,pb]=b(.01*ones(10740,3,'single'),p+[.01 0 0],v,e,true,.05);
            testCase.verifyEqual(ua,ub); testCase.verifyEqual(pa(20:27),pb(20:27));
            [~,~,packet,stop]=b(.01*ones(10740,3,'single'),p,v,e,true,.1);
            testCase.verifyTrue(stop); testCase.verifyEqual(packet(22),0); % fresh unsafe scan is used
        end
        function coincidentStartAndGoalStillLogsATrajectory(testCase)
            options=testCase.Options; options.GoalState=options.StartState;
            map=testCase.Map; plan=ares.navigation.planPath(map,options);
            file=string(tempname(fullfile(testCase.Root,'tmp')))+".mat";
            save(file,'map','options','plan');
            a=ares.navigation.MapCorrectionController('PlanFile',char(file));
            cleanup=onCleanup(@()release(a)); %#ok<NASGU>
            [~,~,packet,stop]=a([],options.StartState(1:3),[0 0 0],[0 0 0],false,0);
            testCase.verifyFalse(stop); testCase.verifyEqual(packet(26),1);
            [~,~,~,stop]=a([],options.StartState(1:3),[0 0 0],[0 0 0],false,.05);
            testCase.verifyTrue(stop);
        end
        function viewerClearsOnFailureAndRestoresOnRecovery(testCase)
            options=testCase.Options;
            options.StartState=[0 3 -7 0 0 0]; options.GoalState=[2 3 -7 0 0 0];
            options.StartEulerZYXRad=[pi/2 0 0];
            map=testCase.Map; plan=ares.navigation.planPath(map,options);
            file=string(tempname(fullfile(testCase.Root,'tmp')))+".mat";
            save(file,'map','options','plan');
            system=ares.navigation.MapCorrectionController('PlanFile',char(file));
            cleanup=onCleanup(@()release(system)); %#ok<NASGU>
            p=options.StartState(1:3); v=[0 0 0]; e=options.StartEulerZYXRad;
            measured=ares.navigation.virtualLidar(map,p,e);
            testCase.verifyTrue(any(isfinite(measured),'all'));
            shown=system(measured,p,v,e,true,0);
            testCase.verifyEqual(shown,measured);
            [shown,~,packet,stop]=system(measured,p,v,e,false,.05);
            testCase.verifyTrue(all(isnan(shown),'all'));
            testCase.verifyEqual(packet(20:22),[0;1;1]);
            testCase.verifyTrue(any(packet(37:43)>0)); % virtual sectors remain internal
            testCase.verifyFalse(stop);
            shown=system(measured,p,v,e,true,.1);
            testCase.verifyEqual(shown,measured);
            reset(system);
            shown=system(measured,p,v,e,false,0);
            testCase.verifyTrue(all(isnan(shown),'all')); % blackout from first tick
        end
    end
end
function releaseBoth(a,b)
release(a); release(b);
end
