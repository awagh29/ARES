function common = navigationOptions(options)
%NAVIGATIONOPTIONS Share mission validation and deterministic outage scheduling.
common=ares.navigation.defaultOptions();
for name=string(fieldnames(options))'
    if isfield(common,name), common.(name)=options.(name); end
end
common.ControlSampleTimeS=.1;
end
