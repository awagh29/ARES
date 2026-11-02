classdef TestLidarEpisode < matlab.unittest.TestCase
    % Synthetic timelines test interpolation, integrity, and immutability.
    properties
        Folder
        TelemetryPath
    end
    methods (TestMethodSetup)
        function prepareFixture(testCase)
            testCase.Folder = string(tempname);
            mkdir(testCase.Folder);
            testCase.addTeardown(@() rmdir(testCase.Folder, "s"));
            testCase.TelemetryPath = fullfile(testCase.Folder, "fixture_telemetry.mat");
            telemetry = TestLidarEpisode.fixture(); %#ok<NASGU>
            save(testCase.TelemetryPath, "telemetry");
        end
    end
    methods (Test)
        function separatesMissionGoalFromControllerReference(testCase)
            data=load(testCase.TelemetryPath);
            telemetry=data.telemetry;
            telemetry.mission.finalGoalPositionNed_m=[30 35 -7];
            save(testCase.TelemetryPath,"telemetry");
            episode=buildLidarEpisode(testCase.TelemetryPath);
            testCase.verifyEqual(episode.goal.positionNed_m,repmat([30 35 -7],3,1));
            testCase.verifyEqual(episode.reference.desiredPositionNed_m(:,1),[10;10;20]);
            testCase.verifyEqual(episode.goal.source,"fixed final mission goal");
        end
        function rejectsInvalidMissionGoal(testCase)
            data=load(testCase.TelemetryPath);
            telemetry=data.telemetry;
            telemetry.mission.finalGoalPositionNed_m=[NaN 35 -7];
            save(testCase.TelemetryPath,"telemetry");
            testCase.verifyError(@()buildLidarEpisode(testCase.TelemetryPath),"ARES:InvalidMissionGoal");
        end
        function alignsWithoutFutureActionLeakage(testCase)
            episode = buildLidarEpisode(testCase.TelemetryPath);
            testCase.verifyEqual(episode.runId, "fixture-run");
            testCase.verifyEqual(episode.state.positionNed_m(:, 1), [0; 1; 2]);
            testCase.verifyEqual(episode.goal.positionNed_m(:, 1), [10; 10; 20]);
            testCase.verifyEqual(episode.action.current(:, 1), [1; 1; 3]);
            testCase.verifyTrue(all(isnan(episode.action.previous(1, :))));
            testCase.verifyEqual(episode.action.previous(2:end, 1), [1; 1]);
            testCase.verifyEqual(episode.action.previousAvailable, [false; true; true]);
            testCase.verifySize(episode.lidar.ranges_m, [3 4]);
            testCase.verifyTrue(all(episode.lidar.availabilityMask, "all"));
            testCase.verifyFalse(any(episode.lidar.missingMask, "all"));
            testCase.verifyFalse(isfield(episode.state, "attitudeNedRpy_rad"));
            saved = load(fullfile(testCase.Folder, "fixture_telemetry_lidar_episode.mat"));
            testCase.verifyEqual(saved.episode.runId, episode.runId);
        end
        function retainsLastDuplicateStateSample(testCase)
            data = load(testCase.TelemetryPath);
            telemetry = data.telemetry;
            telemetry.signals.position_ned_m = TestLidarEpisode.signal( ...
                [0; 0.5; 0.5; 1], [0 0 -7; 99 0 -7; 2 0 -7; 4 0 -7]);
            save(testCase.TelemetryPath, "telemetry");
            episode = buildLidarEpisode(testCase.TelemetryPath);
            testCase.verifyEqual(episode.state.positionNed_m(:, 1), [0; 2; 4]);
        end
        function rejectsExtrapolation(testCase)
            data = load(testCase.TelemetryPath);
            telemetry = data.telemetry;
            telemetry.signals.velocity_ned_mps.time_s = [0.1; 1];
            save(testCase.TelemetryPath, "telemetry");
            testCase.verifyError(@() buildLidarEpisode(testCase.TelemetryPath), ...
                "ARES:ResamplingOutsideTimeRange");
        end
        function acceptsFloatingPointScanEndpointWithoutChangingTimestamps(testCase)
            data = load(testCase.TelemetryPath);
            telemetry = data.telemetry;
            names = fieldnames(telemetry.signals);
            for index = 1:numel(names)
                telemetry.signals.(names{index}).time_s = ...
                    telemetry.signals.(names{index}).time_s * 32.8;
            end
            scanEnd = 32.800000000000004;
            telemetry.signals.nominal_observation.time_s(end) = scanEnd;
            telemetry.signals.nominal_lidar_valid_return_mask.time_s(end) = scanEnd;
            save(testCase.TelemetryPath, "telemetry");
            episode = buildLidarEpisode(testCase.TelemetryPath);
            testCase.verifyGreaterThan(scanEnd, 32.8);
            testCase.verifyEqual(episode.time_s(end), scanEnd);
            testCase.verifyEqual(episode.state.positionNed_m(end, :), [2 0 -7]);
            testCase.verifyEqual(episode.action.current(end, :), [3 6 9 12]);
            testCase.verifyEqual(episode.metadata.timestampBoundaryTolerance_s, 1e-9);
            testCase.verifyFalse(episode.metadata.extrapolationUsed);
        end
        function rejectsScanEndpointOutsideBoundaryTolerance(testCase)
            data = load(testCase.TelemetryPath);
            telemetry = data.telemetry;
            names = fieldnames(telemetry.signals);
            for index = 1:numel(names)
                telemetry.signals.(names{index}).time_s = ...
                    telemetry.signals.(names{index}).time_s * 32.8;
            end
            scanEnd = 32.8 + 1e-6;
            telemetry.signals.nominal_observation.time_s(end) = scanEnd;
            telemetry.signals.nominal_lidar_valid_return_mask.time_s(end) = scanEnd;
            save(testCase.TelemetryPath, "telemetry");
            testCase.verifyError(@() buildLidarEpisode(testCase.TelemetryPath), ...
                "ARES:ResamplingOutsideTimeRange");
        end
        function rejectsHiddenNativeOutage(testCase)
            data = load(testCase.TelemetryPath);
            telemetry = data.telemetry;
            telemetry.signals.lidar_enabled = TestLidarEpisode.signal( ...
                [0; 0.25; 0.3; 1], [1; 0; 1; 1]);
            save(testCase.TelemetryPath, "telemetry");
            testCase.verifyError(@() buildLidarEpisode(testCase.TelemetryPath), ...
                "ARES:UnexpectedLidarOutage");
        end
        function rejectsMisalignedValidityMask(testCase)
            data = load(testCase.TelemetryPath);
            telemetry = data.telemetry;
            telemetry.signals.nominal_lidar_valid_return_mask.time_s(2) = 0.51;
            save(testCase.TelemetryPath, "telemetry");
            testCase.verifyError(@() buildLidarEpisode(testCase.TelemetryPath), ...
                "ARES:AlignedSignalTimeMismatch");
        end
        function rejectsNonfiniteNativeSignal(testCase)
            data = load(testCase.TelemetryPath);
            telemetry = data.telemetry;
            telemetry.signals.command_thrust_n.values(2) = NaN;
            save(testCase.TelemetryPath, "telemetry");
            testCase.verifyError(@() buildLidarEpisode(testCase.TelemetryPath), ...
                "ARES:InvalidSignalValues");
        end
        function refusesToOverwriteEpisode(testCase)
            output = fullfile(testCase.Folder, "episode.mat");
            buildLidarEpisode(testCase.TelemetryPath, output);
            original = dir(output);
            testCase.verifyError(@() buildLidarEpisode(testCase.TelemetryPath, output), ...
                "ARES:EpisodeAlreadyExists");
            current = dir(output);
            testCase.verifyEqual(current.bytes, original.bytes);
            testCase.verifyEqual(current.datenum, original.datenum);
        end
        function interpolatesAnglesAcrossWrap(testCase)
            data = load(testCase.TelemetryPath);
            telemetry = data.telemetry;
            telemetry.signals.attitude_ned_rpy_rad = TestLidarEpisode.signal( ...
                [0; 1], [0 0 deg2rad(179); 0 0 deg2rad(-179)]);
            save(testCase.TelemetryPath, "telemetry");
            episode = buildLidarEpisode(testCase.TelemetryPath);
            testCase.verifyEqual(abs(episode.state.attitudeNedRpy_rad(2, 3)), ...
                pi, AbsTol=1e-12);
        end
        function rejectsInvalidNoReturnEncoding(testCase)
            data = load(testCase.TelemetryPath);
            telemetry = data.telemetry;
            telemetry.signals.nominal_observation.values(1, 4) = 0;
            save(testCase.TelemetryPath, "telemetry");
            testCase.verifyError(@() buildLidarEpisode(testCase.TelemetryPath), ...
                "ARES:InvalidNoReturnEncoding");
        end
    end
    methods (Static, Access=private)
        function telemetry = fixture()
            telemetry.schemaVersion = "2.0.0";
            telemetry.runId = "fixture-run";
            telemetry.coordinateFrame = "NED";
            telemetry.timeBase = "simulation-time";
            telemetry.lidarMetadata.gridSize = [2 2];
            telemetry.lidarMetadata.maximumRangeM = 7;
            telemetry.lidarMetadata.channelOrder = "elevation-index-fastest-then-azimuth-index";
            scanTime = [0; 0.5; 1];
            telemetry.signals.nominal_observation = TestLidarEpisode.signal( ...
                scanTime, repmat([1 2 3 7], 3, 1));
            telemetry.signals.nominal_lidar_valid_return_mask = TestLidarEpisode.signal( ...
                scanTime, repmat([true true true false], 3, 1));
            telemetry.signals.position_ned_m = TestLidarEpisode.signal( ...
                [0; 1], [0 0 -7; 2 0 -7]);
            telemetry.signals.velocity_ned_mps = TestLidarEpisode.signal( ...
                [0; 1], [2 0 0; 2 0 0]);
            telemetry.signals.desired_position_ned_m = TestLidarEpisode.signal( ...
                [0; 0.75; 1], [10 0 -7; 20 0 -7; 20 0 -7]);
            telemetry.signals.lidar_enabled = TestLidarEpisode.signal([0; 1], [1; 1]);
            names = ["command_roll_rad", "command_pitch_rad", ...
                "command_yaw_rate_radps", "command_thrust_n"];
            for index = 1:numel(names)
                telemetry.signals.(names(index)) = TestLidarEpisode.signal( ...
                    [0; 0.75; 1], [1; 2; 3] * index);
            end
        end
        function signal = signal(time_s, values)
            signal.time_s = time_s;
            signal.values = values;
        end
    end
end
