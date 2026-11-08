classdef TestSaitsInference < matlab.unittest.TestCase
    % Native inference and online sensor-source selection without a MEX build.
    properties
        ArtifactPath
        Model
        Fixtures
    end

    methods (TestClassSetup)
        function loadActualArtifact(testCase)
            root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,"src")));
            testCase.ArtifactPath = fullfile(root,"data","models","saits_endpoint","model.mat");
            testCase.assertTrue(isfile(testCase.ArtifactPath), ...
                "Export the trained SAITS checkpoint before running inference tests.");
            artifact = load(testCase.ArtifactPath,"model","fixtures");
            testCase.Model = artifact.model;
            testCase.Fixtures = artifact.fixtures;
        end
    end

    methods (Test)
        function nativePredictionMatchesExportedPythonEndpoints(testCase)
            fixtures = testCase.Fixtures;
            for index = 1:size(fixtures.rawFeatures,3)
                raw = fixtures.rawFeatures(:,:,index);
                observed = fixtures.observedMask(:,:,index);
                [physical,normalized,detail] = ares.inference.predictSaits( ...
                    testCase.Model,raw,observed);
                testCase.verifyEqual(physical,fixtures.predictionPhysical(index,:), ...
                    "AbsTol",single(1e-4));
                testCase.verifyEqual(normalized,fixtures.predictionNormalized(index,:), ...
                    "AbsTol",single(1e-4));
                testCase.verifyEqual(detail.stage1Normalized,fixtures.stage1Normalized(index,:), ...
                    "AbsTol",single(1e-4));
                testCase.verifyEqual(detail.stage2Normalized,fixtures.stage2Normalized(index,:), ...
                    "AbsTol",single(1e-4));
                raw(~observed) = single(NaN);
                fromNaN = ares.inference.predictSaits(testCase.Model,raw);
                testCase.verifyEqual(fromNaN,physical,"AbsTol",single(1e-6));
            end
        end

        function outageOracleCloudCannotAffectFeaturesOrReconstruction(testCase)
            first = testCase.newSystem();
            second = testCase.newSystem();
            measured = TestSaitsInference.cloudAtRange(3);
            oracleNear = TestSaitsInference.cloudAtRange(0.5);
            oracleAbsent = NaN(10740,3,"single");
            for tick = 0:52
                time = tick/10;
                enabled = tick<35;
                if enabled
                    cloudA = measured;
                    cloudB = measured;
                else
                    cloudA = oracleNear;
                    cloudB = oracleAbsent;
                end
                a = TestSaitsInference.call(first,cloudA,enabled,time);
                b = TestSaitsInference.call(second,cloudB,enabled,time);
                testCase.verifyEqual(a.features,b.features);
                testCase.verifyEqual(a.mask,b.mask);
                testCase.verifyEqual(a.predicted,b.predicted);
                testCase.verifyEqual(a.sectors,b.sectors);
                testCase.verifyTrue(isequaln(a.cloud,b.cloud));
                testCase.verifyEqual(a.used,b.used);
                testCase.verifyEqual(a.ready,b.ready);
                if ~enabled
                    testCase.verifyFalse(a.available);
                    testCase.verifyFalse(any(a.mask(1:14)));
                    testCase.verifyEqual(a.features(1:14),zeros(14,1,"single"));
                    testCase.verifyTrue(all(a.mask(15:31)));
                    testCase.verifyEqual(a.ready,tick>=49);
                    testCase.verifyEqual(a.used,tick>=49);
                    if tick<49
                        testCase.verifyTrue(all(isnan(a.cloud),"all"));
                        testCase.verifyEqual(a.reason,uint8(1));
                    else
                        testCase.verifyTrue(a.valid);
                        testCase.verifyEqual(a.reason,uint8(3));
                    end
                end
            end
            [position,~,~,~] = TestSaitsInference.motion(5.2);
            [lastHealthy,~,~,~] = TestSaitsInference.motion(3.4);
            testCase.verifyEqual(a.features(15:17),single(position-lastHealthy).', ...
                "AbsTol",single(1e-6));
            testCase.verifyEqual(a.features(31),single(5.2-3.4),"AbsTol",single(1e-6));
        end

        function repeatedTickDoesNotAdvanceHistory(testCase)
            system = testCase.newSystem();
            measured = TestSaitsInference.cloudAtRange(3);
            initial = TestSaitsInference.call(system,measured,true,0);
            for repetition = 1:60
                repeated = TestSaitsInference.call(system,NaN(10740,3,"single"),false,0);
                testCase.verifyEqual(repeated.features,initial.features);
                testCase.verifyEqual(repeated.available,initial.available);
                testCase.verifyEqual(repeated.mask,initial.mask);
                testCase.verifyEqual(repeated.cloud,initial.cloud);
                testCase.verifyFalse(repeated.ready);
            end
            for tick = 1:48
                output = TestSaitsInference.call(system,measured,false,tick/10);
                testCase.verifyFalse(output.ready);
                testCase.verifyFalse(output.used);
            end
            output = TestSaitsInference.call(system,measured,false,4.9);
            testCase.verifyTrue(output.ready);
            testCase.verifyTrue(output.used);
            testCase.verifyTrue(output.valid);
        end

        function healthyRecoveryPassesCloudAndResetsMotionAge(testCase)
            system = testCase.newSystem();
            measured = TestSaitsInference.cloudAtRange(3);
            for tick = 0:52
                output = TestSaitsInference.call(system,measured,tick<35,tick/10);
            end
            testCase.verifyTrue(output.used);
            recoveredCloud = TestSaitsInference.cloudAtRange(4);
            recovered = TestSaitsInference.call(system,recoveredCloud,true,5.3);
            testCase.verifyEqual(recovered.cloud,recoveredCloud);
            testCase.verifyFalse(recovered.used);
            testCase.verifyTrue(recovered.ready);
            testCase.verifyTrue(recovered.available);
            testCase.verifyEqual(recovered.reason,uint8(0));
            testCase.verifyTrue(all(recovered.mask));
            testCase.verifyEqual(recovered.features(15:17),zeros(3,1,"single"));
            testCase.verifyEqual(recovered.features(30:31),single([1;0]));
            expectedSectors = ares.inference.sectorFeatures(recoveredCloud);
            testCase.verifyEqual(recovered.sectors,expectedSectors.');
            [~,~,eulerZYX,~] = TestSaitsInference.motion(5.3);
            rpy = eulerZYX([3 2 1]);
            expectedOrientation = single(reshape([sin(rpy);cos(rpy)],6,1));
            testCase.verifyEqual(recovered.features(21:26),expectedOrientation);
        end

        function resetDiscardsReadyHistoryAndHealthyPredecessor(testCase)
            system = testCase.newSystem();
            measured = TestSaitsInference.cloudAtRange(3);
            for tick = 0:49
                output = TestSaitsInference.call(system,measured,tick<35,tick/10);
            end
            testCase.verifyTrue(output.ready);
            testCase.verifyTrue(output.used);
            reset(system);
            restarted = TestSaitsInference.call(system,measured,false,0);
            testCase.verifyFalse(restarted.ready);
            testCase.verifyFalse(restarted.used);
            testCase.verifyFalse(restarted.valid);
            testCase.verifyFalse(restarted.available);
            testCase.verifyEqual(restarted.reason,uint8(1));
            testCase.verifyTrue(all(isnan(restarted.cloud),"all"));
            testCase.verifyEqual(restarted.features(15:17),zeros(3,1,"single"));
            testCase.verifyEqual(restarted.features(31),single(0));
            testCase.verifyEqual(restarted.predicted,zeros(14,1,"single"));
        end

        function sectorFeaturesUseReversedSensorBearingAndAllElevations(testCase)
            points = NaN(10740,3,"single");
            azimuth = linspace(-179,179,716);
            bearing = -azimuth;
            edges = linspace(-60,60,8);
            expectedDistance = zeros(1,7,"single");
            expectedFraction = zeros(1,7,"single");
            for sector = 1:7
                center = (edges(sector)+edges(sector+1))/2;
                [~,column] = min(abs(bearing-center));
                radius = 1+0.5*sector;
                point = single([radius*cosd(azimuth(column)), ...
                    radius*sind(azimuth(column)),0]);
                points(8+(column-1)*15,:) = point;
                expectedDistance(sector) = single(norm(double(point)));
                columns = bearing>=edges(sector) & ...
                    (bearing<edges(sector+1) | (sector==7 & bearing<=edges(sector+1)));
                expectedFraction(sector) = single(1/(15*nnz(columns)));
            end
            features = ares.inference.sectorFeatures(points);
            testCase.verifyEqual(features(1:7),expectedDistance,"AbsTol",single(1e-6));
            testCase.verifyEqual(features(8:14),expectedFraction,"AbsTol",single(1e-8));
        end

        function noReturnEncodingDoesNotBecomeSensorLoss(testCase)
            expected = single([7*ones(1,7),zeros(1,7)]);
            testCase.verifyEqual(ares.inference.sectorFeatures(NaN(10740,3,"single")),expected);
            testCase.verifyEqual(ares.inference.sectorFeatures(zeros(10740,3,"single")),expected);
            system = testCase.newSystem();
            measuredNoReturns = NaN(10740,3,"single");
            output = TestSaitsInference.call(system,measuredNoReturns,true,0);
            testCase.verifyTrue(output.available);
            testCase.verifyTrue(all(output.mask));
            testCase.verifyEqual(output.features(1:14),expected.');
            testCase.verifyFalse(output.used);
            testCase.verifyEqual(output.reason,uint8(0));
        end

        function proxyExpandsPositiveReturnsAndGuardsUnknownDirections(testCase)
            prediction = single([3*ones(1,7),0.2*ones(1,7)]);
            [points,controller] = ares.inference.sectorProxyCloud(prediction);
            testCase.verifySize(points,[10740 3]);
            testCase.verifyEqual(controller,prediction);
            ranges = reshape(sqrt(sum(double(points).^2,2)),15,716);
            bearing = -linspace(-179,179,716);
            forward = bearing>=-60 & bearing<=60;
            testCase.verifyEqual(ranges(:,forward),3*ones(15,nnz(forward)),"AbsTol",2e-6);
            testCase.verifyEqual(ranges(:,~forward),6.95*ones(15,nnz(~forward)),"AbsTol",2e-6);
            % The adapter fills a whole sector for any positive return fraction.
            testCase.verifyEqual(ares.inference.sectorFeatures(points), ...
                single([3*ones(1,7),ones(1,7)]),"AbsTol",single(2e-6));
        end

        function proxyClipsFiniteOutliersAndKeepsNoReturnsAbsent(testCase)
            prediction = single([-9,0.05,4,9,7,3,2,-2,2,0.001,0.5,0.5,0,1.1]);
            [points,controller] = ares.inference.sectorProxyCloud(prediction);
            testCase.verifyEqual(controller,single([0.201,0.201,4,7,7,3,2,0,1,0.001,0.5,0.5,0,1]));
            features = ares.inference.sectorFeatures(points);
            testCase.verifyEqual(features,single([7,0.201,4,7,7,7,2,0,1,1,0,0,0,1]), ...
                "AbsTol",single(2e-6));
            prediction(1) = Inf;
            testCase.verifyError(@()ares.inference.sectorProxyCloud(prediction), ...
                "ARES:InvalidSaitsPrediction");
        end
    end

    methods (Access=private)
        function system = newSystem(testCase)
            system = ares.inference.SaitsLidarReconstruction( ...
                "ArtifactPath",testCase.ArtifactPath,"UseMex",false);
            testCase.addTeardown(@()release(system));
        end
    end

    methods (Static,Access=private)
        function output = call(system,cloud,enabled,time)
            [position,velocity,eulerZYX,angular] = TestSaitsInference.motion(time);
            values = cell(1,11);
            [values{:}] = step(system,cloud,position,velocity,eulerZYX,angular,enabled,time);
            output = struct("cloud",values{1},"used",values{2},"ready",values{3}, ...
                "valid",values{4},"reason",values{5},"elapsed",values{6}, ...
                "predicted",values{7},"sectors",values{8},"features",values{9}, ...
                "mask",values{10},"available",values{11});
        end

        function [position,velocity,eulerZYX,angular] = motion(time)
            position = [0.5*time,0.7*time,-7+0.02*sin(time)];
            velocity = [0.5,0.7,0.02*cos(time)];
            eulerZYX = [0.1*time,0.04*cos(time),0.03*sin(time)];
            angular = [0.03*cos(time),-0.04*sin(time),0.1];
        end

        function cloud = cloudAtRange(radius)
            azimuth = linspace(-179,179,716);
            elevation = linspace(-15,15,15).';
            x = single(radius*(cosd(elevation)*cosd(azimuth)));
            y = single(radius*(cosd(elevation)*sind(azimuth)));
            z = single(radius*(sind(elevation)*ones(1,716)));
            cloud = [x(:),y(:),z(:)];
        end
    end
end
