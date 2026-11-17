classdef TestDenseSaitsDisabledScenario < matlab.unittest.TestCase
    % Schedule, telemetry proof and cold-start oracle isolation; no plant run.
    properties
        Root
        ArtifactPath = ""
    end
    methods (TestClassSetup)
        function setupPaths(testCase)
            testCase.Root=string(fileparts(fileparts(fileparts(mfilename("fullpath")))));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(testCase.Root,"src")));
            pointer=fullfile(testCase.Root,"data","models","saits_dense_single_episode", ...
                "f9ab72ef-e1d7-4479-8faa-03bb5b36776c","deployment.json");
            if isfile(pointer)
                deployment=jsondecode(fileread(pointer));
                testCase.ArtifactPath=string(deployment.model_path);
            end
        end
    end
    methods (Test)
        function disabledScheduleHasNoHealthySampleOrRecovery(testCase)
            [schedule,intervals,fault]=ares.inference.denseSaitsLidarScenario(60,"disabled");
            testCase.verifyEqual(double(schedule.Time),[0;60]);
            testCase.verifyEqual(double(schedule.Data),[0;0]);
            testCase.verifyEqual(intervals,[0 60]);
            testCase.verifyEqual(interp1(double(schedule.Time),double(schedule.Data),(0:.1:60)',"previous"),zeros(601,1));
            testCase.verifyEqual(fault.initialHealthyTime_s,0);
            testCase.verifyEqual(fault.finalHealthyTime_s,0);
            testCase.verifyEqual(fault.numberOfOutages,1);
            testCase.verifyEmpty(fault.randomSeed);
        end
        function intermittentScheduleIsUnchanged(testCase)
            [actual,intervals,fault]=ares.inference.denseSaitsLidarScenario(60);
            [expected,originalIntervals,originalFault]=ares.inference.createIntermittentSchedule(60);
            testCase.verifyEqual(actual.Time,expected.Time);
            testCase.verifyEqual(actual.Data,expected.Data);
            testCase.verifyEqual(intervals,originalIntervals);
            testCase.verifyEqual(fault.randomSeed,originalFault.randomSeed);
        end
        function disabledProofCountsOneContinuousOutageAndEveryReconstruction(testCase)
            schedule=ares.inference.denseSaitsLidarScenario(2,"disabled");
            telemetry=TestDenseSaitsDisabledScenario.syntheticTelemetry(schedule,2);
            [scan,proof]=ares.inference.denseSaitsScanTable(telemetry,schedule,"disabled-fixture","disabled");
            testCase.verifyTrue(proof.instrumentationChecksPassed);
            testCase.verifyEqual(proof.observedOutageCount,1);
            testCase.verifyEqual(proof.observedRecoveryCount,0);
            testCase.verifyEqual(proof.enabledLidarScanCount,0);
            testCase.verifyEqual(proof.reconstructionUsedScanCount,height(scan));
            testCase.verifyTrue(proof.firstScanReconstructed && proof.allScansReconstructed && proof.noMeasuredLidarObserved);
            testCase.verifyEqual(scan.OutageAgeS,scan.TimeS);
        end
        function proofRejectsMeasuredLidarOrOracleLeak(testCase)
            schedule=ares.inference.denseSaitsLidarScenario(2,"disabled");
            telemetry=TestDenseSaitsDisabledScenario.syntheticTelemetry(schedule,2);
            telemetry.signals.saits_dense_input_observed_mask.values(1,1)=true;
            telemetry.signals.saits_dense_input_features.values(1,1)=3;
            telemetry.signals.saits_dense_measured_available.values(1)=true;
            [~,proof]=ares.inference.denseSaitsScanTable(telemetry,schedule,"leak-fixture","disabled");
            testCase.verifyFalse(proof.instrumentationChecksPassed);
            testCase.verifyFalse(proof.noMeasuredLidarObserved);
        end
        function intermittentProofStillRequiresThreeOutagesAndRecoveries(testCase)
            schedule=ares.inference.denseSaitsLidarScenario(21,"intermittent");
            telemetry=TestDenseSaitsDisabledScenario.syntheticTelemetry(schedule,21);
            [~,proof]=ares.inference.denseSaitsScanTable(telemetry,schedule,"intermittent-fixture");
            testCase.verifyTrue(proof.instrumentationChecksPassed);
            testCase.verifyEqual(proof.observedOutageCount,3);
            testCase.verifyEqual(proof.observedRecoveryCount,3);
        end
        function coldStartNeverReadsOracleOrRecyclesPredictions(testCase)
            testCase.assumeTrue(isfile(testCase.ArtifactPath),"Train/export the dense SAITS model for native inference checks.");
            artifact=load(testCase.ArtifactPath,"model");
            first=ares.inference.DenseSaitsLidarReconstruction("ArtifactPath",char(testCase.ArtifactPath));
            second=ares.inference.DenseSaitsLidarReconstruction("ArtifactPath",char(testCase.ArtifactPath));
            cleanup=onCleanup(@()releaseBoth(first,second)); %#ok<NASGU>
            for k=0:double(artifact.model.nSteps)+1
                a=TestDenseSaitsDisabledScenario.step(first,single(.05)*ones(10740,3,"single"),k*.1);
                b=TestDenseSaitsDisabledScenario.step(second,NaN(10740,3,"single"),k*.1);
                testCase.verifyTrue(a.used && a.ready && a.valid && a.native);
                testCase.verifyFalse(a.available);
                testCase.verifyEqual(a.reason,uint8(3));
                testCase.verifyEqual(a.dense,b.dense,"AbsTol",single(1e-6));
                testCase.verifyEqual(a.features(1:10740),zeros(10740,1,"single"));
                testCase.verifyFalse(any(a.mask(1:10740)));
                testCase.verifyTrue(all(a.mask(10741:end)));
                if k==0
                    % Independent cold endpoint: nothing measured except the
                    % current motion variables. No hidden historical seed.
                    raw=zeros(double(artifact.model.nSteps),10752,"single");
                    observed=false(size(raw));
                    raw(end,:)=a.features.'; observed(end,:)=a.mask.';
                    expected=ares.inference.predictDenseSaits(artifact.model,raw,observed);
                    testCase.verifyEqual(a.dense,expected(:),"AbsTol",single(1e-6));
                end
            end
            duplicate=TestDenseSaitsDisabledScenario.step(first,NaN(10740,3,"single"),k*.1);
            testCase.verifyEqual(duplicate.dense,a.dense);
            reset(first);
            restarted=TestDenseSaitsDisabledScenario.step(first,NaN(10740,3,"single"),0);
            testCase.verifyEqual(restarted.dense,expected(:),"AbsTol",single(1e-6));
        end
    end
    methods (Static,Access=private)
        function value=step(system,cloud,time)
            [~,value.used,value.ready,value.valid,value.reason,~,~,~,value.features,value.mask,value.available,value.native,value.dense]= ...
                system(cloud,[0 0 -7],[0 0 0],[0 0 0],[0 0 0],false,time);
        end
        function telemetry=syntheticTelemetry(schedule,stopTime)
            time=(0:.1:stopTime)'; n=numel(time);
            available=interp1(double(schedule.Time),double(schedule.Data),time,"previous")>=.5;
            signal=@(values)struct("time_s",time,"values",values);
            signals=struct();
            signals.nominal_observation=signal(single(7)*ones(n,10740,"single"));
            signals.nominal_lidar_valid_return_mask=signal(false(n,10740));
            signals.lidar_enabled=signal(available);
            signals.saits_dense_reconstruction_used=signal(~available);
            signals.saits_dense_model_ready=signal(true(n,1));
            signals.saits_dense_prediction_valid=signal(~available);
            signals.saits_dense_fallback_reason=signal(3*double(~available));
            signals.saits_dense_inference_seconds=signal(zeros(n,1));
            signals.saits_dense_predicted_sectors=signal(zeros(n,14));
            signals.saits_dense_controller_sectors=signal(zeros(n,14));
            input=zeros(n,10752,"single"); input(available,1:10740)=7;
            observed=true(n,10752); observed(~available,1:10740)=false;
            signals.saits_dense_input_features=signal(input);
            signals.saits_dense_input_observed_mask=signal(observed);
            signals.saits_dense_measured_available=signal(available);
            signals.saits_dense_backend_native=signal(true(n,1));
            signals.saits_dense_guidance_enabled=signal(true(n,1));
            for name=["saits_dense_selected_reference_ned_m","saits_dense_waypoint_reference_ned_m", ...
                    "saits_dense_reference_delta_ned_m","position_ned_m","velocity_ned_mps"]
                signals.(name)=signal(zeros(n,3));
            end
            for name=["command_roll_rad","command_pitch_rad","command_yaw_rate_radps","command_thrust_n"]
                signals.(name)=signal(zeros(n,1));
            end
            telemetry=struct("signals",signals);
        end
    end
end

function releaseBoth(first,second)
release(first); release(second);
end
