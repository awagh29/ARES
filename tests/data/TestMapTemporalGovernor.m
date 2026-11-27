classdef TestMapTemporalGovernor < matlab.unittest.TestCase
    % Check braking on actual momentum, transform geometry and terminal gating.
    methods (TestClassSetup)
        function locate(~)
            root=string(fileparts(fileparts(fileparts(mfilename('fullpath')))));
            addpath(fullfile(root,'src'));
        end
    end
    methods (Test)
        function sensorMountAndCapsule(test)
            offsets=ares.mapTemporal.cloudOffsets([3 2 0;NaN NaN NaN],[pi/2 0 0]);
            test.verifyEqual(offsets,[2 3 -.4],'AbsTol',1e-12);
            d=ares.mapTemporal.corridorDistance([4 0 0],[1 0 0],1,7);
            test.verifyEqual(d,3);
            test.verifyEqual(ares.mapTemporal.corridorDistance([4 2 0],[1 0 0],1,7),7);
            test.verifyEqual(ares.mapTemporal.corridorDistance([-4 0 0],[1 0 0],1,7),7);
            test.verifyEqual(ares.mapTemporal.corridorDistance([-.5 0 0],[1 0 0],1,7),7);
        end
        function approachesBeforeOldLatchAndRejectsBlockedGoal(test)
            options=ares.mapTemporal.defaultOptions(); options.GoalState=[3 0 -7 0 0 0]; %#ok<NASGU>
            file=string(tempname)+'.mat'; save(file,'options'); clean=onCleanup(@()delete(file)); %#ok<NASGU>
            hold=ares.mapTemporal.GoalHoldReference('PlanFile',file);
            nominal=single([1;1;-7]); cloud=NaN(10740,3,'single');
            cloud(1,:)=[1.5 0 -.4]; % physical mount maps this to same-altitude obstacle
            [ref,~,active]=hold(nominal,single(0),single([0;0;-7]),single([0;0;0]),cloud);
            test.verifyFalse(active); test.verifyEqual(ref,nominal);
            cloud(:)=NaN;
            [ref,~,active]=hold(nominal,single(0),single([0;0;-7]),single([0;0;0]),cloud);
            test.verifyTrue(active); test.verifyEqual(ref,single([3;0;-7]));
            cloud(1,:)=[1.5 0 -.4];
            [ref,~,active]=hold(nominal,single(0),single([0;0;-7]),single([0;0;0]),cloud);
            test.verifyFalse(active); test.verifyEqual(ref,nominal); release(hold);
        end
        function limitsSpeedBrakesMomentumAndTapersAtGoal(test)
            options=ares.mapTemporal.defaultOptions(); %#ok<NASGU>
            options.EnableLocalPlanner=false;
            file=string(tempname)+'.mat'; save(file,'options'); clean=onCleanup(@()delete(file)); %#ok<NASGU>
            gov=ares.mapTemporal.SpeedGovernor('PlanFile',file);
            cloud=NaN(10740,3,'single'); z=single(zeros(3,1)); p=single([0;0;-7]);
            for k=1:30, [command,~]=gov(single([4;0;0]),p,z,z,cloud,false); end
            test.verifyLessThanOrEqual(norm(command),options.CruiseSpeedMps+1e-6);
            % Requested turn is clear, but current forward velocity cannot stop.
            cloud(1,:)=[2 0 -.4];
            [command,packet]=gov(single([0;1;0]),p,single([2;0;0]),z,cloud,false);
            test.verifyEqual(command,z); test.verifyEqual(packet(10),1);
            reset(gov); cloud(:)=NaN;
            p=single(options.GoalState(1:3)')+single([.1;0;0]);
            [command,~]=gov(single([-1;0;0]),p,z,z,cloud,true);
            test.verifyLessThanOrEqual(norm(command),options.GoalApproachGain*.1+1e-6);
            release(gov);
        end
        function legacySwitchAndValidation(test)
            o=ares.mapTemporal.defaultOptions(); o.EnableReferenceGovernor=false;
            test.verifyFalse(ares.mapTemporal.validateOptions(o).EnableReferenceGovernor);
            o.GovernorLatencyS=NaN;
            test.verifyError(@()ares.mapTemporal.validateOptions(o),'ARES:TemporalGovernorOption');
        end
    end
end
