function features = sectorFeatures(sensorPoints)
%SECTORFEATURES The exact training geometry and 14 clean-sector features.
% Sensor FLU, 180-degree roll mounting: body bearing = -sensor azimuth.
assert(isequal(size(sensorPoints),[10740 3]),"ARES:InvalidSaitsCloud", ...
    "Expected the elevation-fastest 15-by-716 organized cloud.");
points = double(sensorPoints);
ranges = sqrt(sum(points.^2,2));
valid = all(isfinite(points),2) & isfinite(ranges) & ranges>0 & ranges<=7+7e-6;
ranges(valid) = min(ranges(valid),7);
ranges(~valid) = 7;
rangeGrid = reshape(ranges,15,716);
validGrid = reshape(valid,15,716);
bearing = -linspace(-179,179,716);
edges = linspace(-60,60,8);
features = [7*ones(1,7),zeros(1,7)];
for k = 1:7
    columns = bearing>=edges(k) & (bearing<edges(k+1) | (k==7 & bearing<=edges(k+1)));
    mask = validGrid(:,columns);
    distance = rangeGrid(:,columns);
    if any(mask,"all"), features(k) = min(distance(mask)); end
    features(7+k) = nnz(mask)/numel(mask);
end
features = single(features);
end
