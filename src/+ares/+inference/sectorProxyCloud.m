function [points, controllerSectors] = sectorProxyCloud(prediction)
%SECTORPROXYCLOUD Explicit coarse forward-sector to VFH sensor-XYZ adapter.
% This is not dense-beam reconstruction. A predicted return fills its whole
% angular sector conservatively; return fraction is not model confidence.
% Unmodelled side/rear directions receive a static 6.95-m unknown-space guard.
assert(numel(prediction)==14 && all(isfinite(prediction)), ...
    "ARES:InvalidSaitsPrediction","Expected fourteen finite physical predictions.");
prediction = single(reshape(prediction,1,14));
controllerSectors = [min(max(prediction(1:7),single(.201)),single(7)), ...
    min(max(prediction(8:14),single(0)),single(1))];
azimuth = linspace(-179,179,716);
elevation = linspace(-15,15,15).';
bearing = -azimuth;
edges = linspace(-60,60,8);
distances = single(6.95)*ones(15,716,"single");
for k = 1:7
    columns = bearing>=edges(k) & (bearing<edges(k+1) | (k==7 & bearing<=edges(k+1)));
    if controllerSectors(7+k)>0 && controllerSectors(k)<7
        distances(:,columns) = controllerSectors(k);
    else
        distances(:,columns) = NaN;
    end
end
% Points remain in the physical sensor frame; VFH applies its existing mount.
x = distances.*single(cosd(elevation)*cosd(azimuth));
y = distances.*single(cosd(elevation)*sind(azimuth));
z = distances.*single(sind(elevation)*ones(1,716));
points = [x(:),y(:),z(:)];
end
