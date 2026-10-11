guidanceModelForBusSetup = multirotor;

environmentTemplateForBusSetup = ...
    environment(guidanceModelForBusSetup);

controlTemplateForBusSetup = ...
    control(guidanceModelForBusSetup);

environmentBusForSetup = localBusFromStruct( ...
    environmentTemplateForBusSetup);

controlBusForSetup = localBusFromStruct( ...
    controlTemplateForBusSetup);

assignin( ...
    "base", ...
    "uavGuidanceEnvironmentBus", ...
    environmentBusForSetup);

assignin( ...
    "base", ...
    "MultirotorGuidanceEnvironmentBus", ...
    environmentBusForSetup);

% Fields are ordered:
% Roll, Pitch, YawRate, Thrust.
assignin( ...
    "base", ...
    "uavGuidanceControlBus", ...
    controlBusForSetup);

% UAV Scenario platform-motion structure.
platformMotionTemplateForBusSetup = struct();

platformMotionTemplateForBusSetup.PlatformID = 0;
platformMotionTemplateForBusSetup.Position = zeros(1, 3);
platformMotionTemplateForBusSetup.Orientation = [1 0 0 0];
platformMotionTemplateForBusSetup.Velocity = zeros(1, 3);
platformMotionTemplateForBusSetup.Acceleration = zeros(1, 3);
platformMotionTemplateForBusSetup.AngularVelocity = zeros(1, 3);

platformMotionBusForSetup = localBusFromStruct( ...
    platformMotionTemplateForBusSetup);

assignin( ...
    "base", ...
    "PlatformMotionBus", ...
    platformMotionBusForSetup);

fprintf( ...
    "ARES bus objects initialized: " + ...
    "environment, control, and platform motion.\n");

clear guidanceModelForBusSetup
clear environmentTemplateForBusSetup
clear controlTemplateForBusSetup
clear environmentBusForSetup
clear controlBusForSetup
clear platformMotionTemplateForBusSetup
clear platformMotionBusForSetup

function busObject = localBusFromStruct(template)

    elementNames = fieldnames(template);
    numberOfElements = numel(elementNames);

    busElements(numberOfElements, 1) = ...
        Simulink.BusElement;

    for elementIndex = 1:numberOfElements

        elementName = elementNames{elementIndex};
        elementValue = template.(elementName);

        busElement = Simulink.BusElement;
        busElement.Name = elementName;

        if isscalar(elementValue)
            busElement.Dimensions = 1;
        else
            busElement.Dimensions = size(elementValue);
        end

        if islogical(elementValue)
            busElement.DataType = "boolean";
        else
            busElement.DataType = class(elementValue);
        end

        busElement.DimensionsMode = "Fixed";

        if isreal(elementValue)
            busElement.Complexity = "real";
        else
            busElement.Complexity = "complex";
        end

        busElements(elementIndex) = busElement;
    end

    busObject = Simulink.Bus;
    busObject.Elements = busElements;
end