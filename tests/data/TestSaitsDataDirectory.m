classdef TestSaitsDataDirectory < matlab.unittest.TestCase
    % Exercise real immutable writers in disposable fixture directories.
    % No Simulink model, trained artifact, or existing data folder is needed.
    properties
        RootDirectory
    end

    methods(TestMethodSetup)
        function useDisposableWorkspace(testCase)
            projectRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                fullfile(projectRoot,"src")));
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture());
            testCase.RootDirectory = string(folder.Folder);
        end
    end

    methods(Test)
        function twoArgumentCallKeepsExistingRawLayout(testCase)
            [manifest,telemetry] = testCase.payload();
            manifestPath = ares.data.saveRuntimeManifest(testCase.RootDirectory,manifest);
            telemetryPath = ares.data.saveRawTelemetry(testCase.RootDirectory,telemetry);
            expectedFolder = fullfile(testCase.RootDirectory,"data","raw",manifest.runId);
            testCase.verifyEqual(manifestPath,fullfile(expectedFolder,"manifest.json"));
            testCase.verifyEqual(telemetryPath,fullfile(expectedFolder, ...
                manifest.simulation.scenarioName+"_telemetry.mat"));
            testCase.verifyRoundTrip(manifestPath,telemetryPath,manifest,telemetry);
            testCase.verifyFalse(isfolder(fullfile(testCase.RootDirectory,"data","saits_intermittent")));
        end

        function saitsDestinationUsesItsOwnUuidFolder(testCase)
            [manifest,telemetry] = testCase.payload();
            manifestPath = ares.data.saveRuntimeManifest( ...
                testCase.RootDirectory,manifest,"saits_intermittent");
            telemetryPath = ares.data.saveRawTelemetry( ...
                testCase.RootDirectory,telemetry,"saits_intermittent");
            expectedFolder = fullfile(testCase.RootDirectory,"data","saits_intermittent",manifest.runId);
            testCase.verifyEqual(manifestPath,fullfile(expectedFolder,"manifest.json"));
            testCase.verifyEqual(string(fileparts(telemetryPath)),expectedFolder);
            testCase.verifyRoundTrip(manifestPath,telemetryPath,manifest,telemetry);
            testCase.verifyFalse(isfolder(fullfile(testCase.RootDirectory,"data","raw")));
        end

        function explicitRawDestinationRemainsSupported(testCase)
            [manifest,telemetry] = testCase.payload();
            manifestPath = ares.data.saveRuntimeManifest(testCase.RootDirectory,manifest,"raw");
            telemetryPath = ares.data.saveRawTelemetry(testCase.RootDirectory,telemetry,"raw");
            testCase.verifyEqual(string(fileparts(manifestPath)), ...
                fullfile(testCase.RootDirectory,"data","raw",manifest.runId));
            testCase.verifyRoundTrip(manifestPath,telemetryPath,manifest,telemetry);
        end

        function completedArtifactsCannotBeOverwritten(testCase)
            [manifest,telemetry] = testCase.payload();
            manifestPath = ares.data.saveRuntimeManifest( ...
                testCase.RootDirectory,manifest,"saits_intermittent");
            telemetryPath = ares.data.saveRawTelemetry( ...
                testCase.RootDirectory,telemetry,"saits_intermittent");
            originalManifest = fileread(manifestPath);
            changedManifest = manifest;
            changedManifest.configuration.testMarker = "would-overwrite-original";
            changedTelemetry = telemetry;
            changedTelemetry.signals.test_value.values(:) = -999;
            testCase.verifyError(@()ares.data.saveRuntimeManifest( ...
                testCase.RootDirectory,changedManifest,"saits_intermittent"),"ARES:RunAlreadyExists");
            testCase.verifyError(@()ares.data.saveRawTelemetry( ...
                testCase.RootDirectory,changedTelemetry,"saits_intermittent"),"ARES:TelemetryAlreadyExists");
            testCase.verifyEqual(fileread(manifestPath),originalManifest);
            saved = load(telemetryPath,"telemetry");
            testCase.verifyEqual(saved.telemetry,telemetry);
        end

        function unsafeSubdirectoriesAreRejectedBeforeWriting(testCase)
            [manifest,telemetry] = testCase.payload();
            unsafe = ["",".","..","../raw","..\raw","safe/child","safe\child", ...
                "C:\outside","/outside","raw.","raw ","CON","nul","COM1","lpt9"];
            for value = unsafe
                testCase.verifyError(@()ares.data.saveRuntimeManifest( ...
                    testCase.RootDirectory,manifest,value),"ARES:InvalidDataSubdirectory");
                testCase.verifyError(@()ares.data.saveRawTelemetry( ...
                    testCase.RootDirectory,telemetry,value),"ARES:InvalidDataSubdirectory");
            end
            testCase.verifyFalse(isfolder(fullfile(testCase.RootDirectory,"data")));
        end

        function telemetryMustUseTheMatchingDestinationManifest(testCase)
            [manifest,telemetry] = testCase.payload();
            ares.data.saveRuntimeManifest(testCase.RootDirectory,manifest,"saits_intermittent");
            testCase.verifyError(@()ares.data.saveRawTelemetry( ...
                testCase.RootDirectory,telemetry),"ARES:MissingManifest");
            testCase.verifyFalse(isfolder(fullfile(testCase.RootDirectory,"data","raw")));
        end

        function mismatchInSavedManifestIdIsRejected(testCase)
            [manifest,telemetry] = testCase.payload();
            manifestPath = ares.data.saveRuntimeManifest( ...
                testCase.RootDirectory,manifest,"saits_intermittent");
            % Corrupt only this test's disposable fixture; production data is untouched.
            changed = manifest;
            changed.runId = "different-run-id";
            TestSaitsDataDirectory.writeFixtureJson(manifestPath,changed);
            testCase.verifyError(@()ares.data.saveRawTelemetry( ...
                testCase.RootDirectory,telemetry,"saits_intermittent"),"ARES:RunIdMismatch");
            testCase.verifyFalse(isfile(fullfile(fileparts(manifestPath), ...
                manifest.simulation.scenarioName+"_telemetry.mat")));
        end

        function schemaMismatchIsRejectedWithoutWritingTelemetry(testCase)
            [manifest,telemetry] = testCase.payload();
            manifestPath = ares.data.saveRuntimeManifest( ...
                testCase.RootDirectory,manifest,"saits_intermittent");
            telemetry.schemaVersion = "incompatible-test-schema";
            testCase.verifyError(@()ares.data.saveRawTelemetry( ...
                testCase.RootDirectory,telemetry,"saits_intermittent"),"ARES:SchemaMismatch");
            testCase.verifyFalse(isfile(fullfile(fileparts(manifestPath), ...
                manifest.simulation.scenarioName+"_telemetry.mat")));
        end
    end

    methods(Access=private)
        function [manifest,telemetry] = payload(~)
            manifest.schemaVersion = "2.0.0";
            manifest.runId = string(java.util.UUID.randomUUID());
            manifest.createdAtUtc = "2026-11-08T00:00:00.000Z";
            manifest.configuration = struct("testMarker","original-fixture");
            manifest.simulation.scenarioName = "saits-writer-test";
            telemetry.schemaVersion = manifest.schemaVersion;
            telemetry.runId = manifest.runId;
            telemetry.signals.test_value = struct("time_s",[0;0.1], ...
                "values",[1;2],"unit","dimensionless","components","test");
        end

        function verifyRoundTrip(testCase,manifestPath,telemetryPath,manifest,telemetry)
            testCase.verifyTrue(isfile(manifestPath));
            testCase.verifyTrue(isfile(telemetryPath));
            restoredManifest = jsondecode(fileread(manifestPath));
            restored = load(telemetryPath,"telemetry");
            testCase.verifyEqual(string(restoredManifest.runId),manifest.runId);
            testCase.verifyEqual(string(restoredManifest.schemaVersion),"2.0.0");
            testCase.verifyEqual(restored.telemetry,telemetry);
            testCase.verifyEqual(string(restoredManifest.runId),restored.telemetry.runId);
            testCase.verifyEqual(string(restoredManifest.schemaVersion),restored.telemetry.schemaVersion);
        end
    end

    methods(Static,Access=private)
        function writeFixtureJson(path,value)
            file = fopen(path,"w","n","UTF-8");
            assert(file~=-1,"ARES:TestFixtureWriteFailed","Cannot write temporary fixture.");
            cleanup = onCleanup(@()fclose(file));
            fprintf(file,"%s",jsonencode(value));
        end
    end
end
