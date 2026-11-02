classdef TestAxisStartPlan < matlab.unittest.TestCase
    methods(Test)
        function planIsReproducibleAndDoesNotChangeGlobalRng(testCase)
            before = rng;
            plan = ares.data.createAxisStartPlan();
            after = rng;
            testCase.verifyEqual(before,after);
            testCase.verifyEqual(plan,ares.data.createAxisStartPlan());
            testCase.verifyNotEqual(plan,ares.data.createAxisStartPlan(7));
        end

        function planCoversExactlyFortyStartsOnEachNedAxis(testCase)
            plan = ares.data.createAxisStartPlan();
            north = plan(plan.Axis=="North",:);
            east = plan(plan.Axis=="East",:);
            testCase.verifyEqual(height(plan),80);
            testCase.verifyEqual(height(north),40);
            testCase.verifyEqual(height(east),40);
            testCase.verifyEqual(north.PositionEastM,zeros(40,1));
            testCase.verifyEqual(east.PositionNorthM,zeros(40,1));
            testCase.verifyEqual(plan.PositionDownM,repmat(-7,80,1));
            testCase.verifyGreaterThan(north.PositionNorthM,zeros(40,1));
            testCase.verifyLessThan(north.PositionNorthM,repmat(30,40,1));
            testCase.verifyGreaterThan(east.PositionEastM,zeros(40,1));
            testCase.verifyLessThan(east.PositionEastM,repmat(35,40,1));
            starts = [plan.PositionNorthM,plan.PositionEastM,plan.PositionDownM];
            testCase.verifyEqual(size(unique(starts,"rows"),1),80);
            testCase.verifyEqual(numel(unique(plan.SimulationSeed)),80);
        end

        function eastAxisExclusionUsesPhysicalEnuMeshCoordinates(testCase)
            cfg = ares.config.defaultSingleUav();
            centers = [0 10;15 15;20 10;15 5;18 22;25 15;20 10];
            plan = ares.data.createAxisStartPlan();
            east = plan.PositionEastM(plan.Axis=="East");
            testCase.verifyTrue(all(east<7.4 | east>12.6));
            position = [plan.PositionNorthM,plan.PositionEastM,plan.PositionDownM];
            minimumDistance = inf(80,1);
            for obstacleIndex=1:size(centers,1)
                delta = max(abs(position(:,1:2)-centers(obstacleIndex,:))-1.5,0);
                minimumDistance = min(minimumDistance,hypot(delta(:,1),delta(:,2)));
            end
            testCase.verifyGreaterThanOrEqual(minimumDistance, ...
                repmat(cfg.vehicle.radiusM+cfg.safety.minimumSurfaceClearanceM,80,1));
            % Every safe North stratum contains one random point.
            testCase.verifyEqual(floor(plan.PositionNorthM(1:40)/30*40),(0:39)');
        end

        function metricsDistinguishSurfaceClearanceFromCenterDistance(testCase)
            cfg = ares.config.defaultSingleUav();
            cfg.mission.initialPositionNed = [0 0 -7];
            cfg.mission.waypointsNed = [0 0 -7;0 4 -7];
            telemetry.signals.position_ned_m = struct( ...
                "time_s",[0;1;2],"values",[0 0 -7;0 2 -7;0 4 -7]);
            metrics = ares.data.evaluateAxisRun(telemetry,cfg,[0 4]);
            testCase.verifyEqual(metrics.minimumObstacleCenterDistanceM,-1.5,"AbsTol",1e-12);
            testCase.verifyEqual(metrics.minimumObstacleClearanceM,-1.6,"AbsTol",1e-12);
            testCase.verifyTrue(metrics.collisionOccurred);
            testCase.verifyTrue(metrics.goalReachedAtTermination);
            testCase.verifyFalse(metrics.missionSuccess);
            testCase.verifyEqual(metrics.totalPathLengthM,4,"AbsTol",1e-12);
        end

        function metricsRespectVerticalObstacleExtent(testCase)
            cfg = ares.config.defaultSingleUav();
            cfg.mission.initialPositionNed = [0 4 -17];
            cfg.mission.waypointsNed = [0 4 -17;0 4 -17];
            telemetry.signals.position_ned_m = struct( ...
                "time_s",[0;1],"values",[0 4 -17;0 4 -17]);
            metrics = ares.data.evaluateAxisRun(telemetry,cfg,[0 4]);
            testCase.verifyEqual(metrics.minimumObstacleCenterDistanceM,2,"AbsTol",1e-12);
            testCase.verifyEqual(metrics.minimumObstacleClearanceM,1.9,"AbsTol",1e-12);
            testCase.verifyFalse(metrics.collisionOccurred);
            testCase.verifyFalse(metrics.altitudeWithinLimits);
        end

        function timeLimitIsNotSuccessfulCompletion(testCase)
            cfg = ares.config.defaultSingleUav();
            telemetry.signals.position_ned_m = struct( ...
                "time_s",[0;cfg.simulation.stopTime], ...
                "values",[cfg.mission.initialPositionNed;cfg.mission.initialPositionNed]);
            metrics = ares.data.evaluateAxisRun(telemetry,cfg,zeros(0,2));
            testCase.verifyTrue(metrics.timedOut);
            testCase.verifyFalse(metrics.missionSuccess);
            testCase.verifyEqual(metrics.terminationReason,"time-limit");
            testCase.verifyEqual(metrics.minimumObstacleClearanceM,inf);
        end

        function invalidTelemetryDoesNotPassMissionChecks(testCase)
            cfg = ares.config.defaultSingleUav();
            telemetry.signals.position_ned_m = struct( ...
                "time_s",[0;1],"values",[0 0 -7;NaN 0 -7]);
            metrics = ares.data.evaluateAxisRun(telemetry,cfg,zeros(0,2));
            testCase.verifyFalse(metrics.telemetryValid);
            testCase.verifyFalse(metrics.missionSuccess);
            testCase.verifyEqual(metrics.terminationReason,"invalid-telemetry");
        end
    end
end
