function signal = timeseriesToSignal(signalSeries, expectedWidth)
arguments
    signalSeries (1,1) timeseries
    expectedWidth (1,1) double {mustBeInteger, mustBePositive}
end

timeValues = double(signalSeries.Time(:));
rawValues = signalSeries.Data;
sampleCount = numel(timeValues);

if sampleCount == 0
    error("ARES:EmptySignal", ...
        "The timeseries contains no samples.");
end

% [channels x 1 x samples].
if size(rawValues, ndims(rawValues)) == sampleCount
    values = reshape(rawValues, [], sampleCount).';

    % Also support ordinary [samples x channels] data.
elseif size(rawValues, 1) == sampleCount
    values = reshape(rawValues, sampleCount, []);

else
    error("ARES:UnsupportedSignalShape", ...
        "Cannot align data size %s with %d timestamps.", ...
        mat2str(size(rawValues)), sampleCount);
end

if size(values, 2) ~= expectedWidth
    error("ARES:UnexpectedSignalWidth", ...
        "Expected %d columns but found %d.", ...
        expectedWidth, size(values, 2));
end

if any(diff(timeValues) < 0)
    error("ARES:InvalidTimestamps", ...
        "Signal timestamps are not monotonic.");
end

if any(~isfinite(timeValues)) || ...
        any(~isfinite(double(values(:))))
    error("ARES:NonfiniteSignal", ...
        "Signal contains NaN or Inf values.");
end

signal.time_s = timeValues;
signal.values = values;
signal.sampleCount = sampleCount;
signal.dataType = string(class(values));

if sampleCount > 1
    signal.nominalSamplePeriod_s = ...
        median(diff(timeValues));
else
    signal.nominalSamplePeriod_s = NaN;
end

end