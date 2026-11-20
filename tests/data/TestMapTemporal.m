classdef TestMapTemporal < matlab.unittest.TestCase
    % Native parity, outage isolation, temporal lifetime and mission bounds.
    properties
        Root
        Artifact
    end
    methods (TestClassSetup)
        function locate(test)
            test.Root=string(fileparts(fileparts(fileparts(mfilename('fullpath')))));
            addpath(fullfile(test.Root,'src'));
            exports=dir(fullfile(test.Root,'results','map_temporal_exports','**','model.mat'));
            test.assumeNotEmpty(exports,'Export a map-temporal checkpoint before runtime tests.');
            [~,index]=max([exports.datenum]); test.Artifact=string(fullfile(exports(index).folder,exports(index).name));
        end
    end
    methods (Test)
        function pytorchParity(test)
            report=ares.mapTemporal.validateArtifact(test.Artifact);
            test.verifyLessThanOrEqual(report.MaximumErrorM,5e-4);
        end
        function missionBounds(test)
            o=ares.mapTemporal.defaultOptions();
            test.verifyEqual(o.WorldBoundsNedM,[-5 40;-5 40;-10 -1]);
            o.StartState(3)=-11;
            test.verifyError(@()ares.mapTemporal.validateOptions(o),'ARES:TemporalStateBounds');
            o.StartState(3)=-7; o.GoalState(4)=1;
            test.verifyError(@()ares.mapTemporal.validateOptions(o),'ARES:MapTerminalVelocity');
        end
        function historyDoesNotObserveHiddenRanges(test)
            a=load(test.Artifact,'model'); m=a.model;
            one=ares.mapTemporal.emptyHistory(m); two=one;
            for k=1:double(m.historySteps)+3
                one=ares.mapTemporal.pushHistory(m,one,NaN(15,716),true(15,716),false(15,716), ...
                    [0 0 -7 0 0 0 0 0 0],[30 35 -7],(k-1)*.1);
                two=ares.mapTemporal.pushHistory(m,two,rand(15,716),false(15,716),false(15,716), ...
                    [0 0 -7 0 0 0 0 0 0],[30 35 -7],(k-1)*.1);
            end
            test.verifyEqual(one,two);
            test.verifyEqual(nnz(one.Present),double(m.historySteps));
            test.verifyEqual(double(one.Ages(end)),(double(m.historySteps)+3)*.1,'AbsTol',1e-5);
            test.verifyFalse(any(one.Summaries,'all'));
        end
        function blackoutCloudIsIsolatedAndResettable(test)
            options=ares.mapTemporal.defaultOptions(); %#ok<NASGU>
            planFile=string(tempname)+'.mat'; save(planFile,'options');
            cleanup=onCleanup(@()delete(planFile)); %#ok<NASGU>
            first=ares.mapTemporal.LidarReconstruction('ArtifactPath',test.Artifact,'PlanFile',planFile);
            second=ares.mapTemporal.LidarReconstruction('ArtifactPath',test.Artifact,'PlanFile',planFile);
            outputs1=cell(1,16); outputs2=outputs1;
            [outputs1{:}]=first(NaN(10740,3),[0 0 -7],[0 0 0],[0 0 0],[0 0 0],0,0);
            [outputs2{:}]=second(999*ones(10740,3),[0 0 -7],[0 0 0],[0 0 0],[0 0 0],0,0);
            test.verifyEqual(outputs1{1},outputs2{1}); test.verifyEqual(outputs1{13},outputs2{13});
            test.verifyTrue(outputs1{2}); test.verifyEqual(outputs1{16}(15),1);
            test.verifyEqual(outputs1{9}(1:10740),zeros(10740,1,'single'));
            test.verifyFalse(any(outputs1{10}(1:10740))); test.verifyFalse(any(outputs1{14}));
            % A duplicate scan does not append history. Reset gives a cold start.
            duplicate=cell(1,16);
            [duplicate{:}]=first(ones(10740,3),[0 0 -7],[0 0 0],[0 0 0],[0 0 0],0,0);
            test.verifyEqual(duplicate,outputs1);
            reset(first);
            [duplicate{:}]=first(ones(10740,3),[0 0 -7],[0 0 0],[0 0 0],[0 0 0],0,0);
            test.verifyEqual(duplicate{13},outputs1{13}); test.verifyEqual(duplicate{16}(15),1);
            release(first); release(second);
        end
        function healthyScanSeedsHistoryAndPassesThrough(test)
            options=ares.mapTemporal.defaultOptions(); %#ok<NASGU>
            planFile=string(tempname)+'.mat'; save(planFile,'options');
            cleanup=onCleanup(@()delete(planFile)); %#ok<NASGU>
            system=ares.mapTemporal.LidarReconstruction('ArtifactPath',test.Artifact,'PlanFile',planFile);
            cloud=NaN(10740,3,'single'); cloud(1,:)=[2 0 0];
            out=cell(1,16);
            [out{:}]=system(cloud,[0 0 -7],[0 0 0],[0 0 0],[0 0 0],1,0);
            test.verifyEqual(out{1},cloud); test.verifyFalse(out{2});
            test.verifyTrue(all(isfinite(out{16})));
            test.verifyEqual(out{16}(15:16),[1;0]); test.verifyEqual(out{13}(1),single(2));
            [out{:}]=system(cloud,[0 0 -7],[0 0 0],[0 0 0],[0 0 0],0,.1);
            test.verifyTrue(out{2}); test.verifyEqual(out{16}(15:16),[2;.1],'AbsTol',1e-6);
            release(system);
        end
        function terminalHoldLatchesAndResets(test)
            options=ares.mapTemporal.defaultOptions(); %#ok<NASGU>
            planFile=string(tempname)+'.mat'; save(planFile,'options');
            cleanup=onCleanup(@()delete(planFile)); %#ok<NASGU>
            hold=ares.mapTemporal.GoalHoldReference('PlanFile',planFile);
            nominal=single([1;2;-7]); goal=single(options.GoalState(1:3)');
            [reference,~,active]=hold(nominal,single(1),single([0;0;-7]),single([.3;0;0]));
            test.verifyEqual(reference,nominal); test.verifyFalse(active);
            [reference,yaw,active]=hold(nominal,single(1),goal+single([.1;0;0]),single([.3;0;0]));
            test.verifyEqual(reference,goal); test.verifyEqual(yaw,single(.3)); test.verifyTrue(active);
            [reference,~,active]=hold(nominal,single(1),goal+single([2;0;0]),single([.6;0;0]));
            test.verifyEqual(reference,goal); test.verifyTrue(active);
            reset(hold);
            [reference,~,active]=hold(nominal,single(1),single([0;0;-7]),single([.3;0;0]));
            test.verifyEqual(reference,nominal); test.verifyFalse(active); release(hold);
        end
    end
end
