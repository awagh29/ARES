function [cloud,clipped] = cnnRangeCloud(ranges)
%CNNRANGECLOUD Use the recorded FLU ray geometry, never the outage oracle cloud.
assert(isequal(size(ranges),[15 716]) && all(isfinite(ranges),"all"), ...
    "ARES:InvalidCnnRanges","Expected finite 15x716 ranges.");
clipped=min(max(single(ranges),single(0)),single(7));
persistent rays
if isempty(rays)
    [elevation,azimuth]=ndgrid(linspace(-15,15,15)*pi/180,linspace(-179,179,716)*pi/180);
    rays=single([cos(elevation(:)).*cos(azimuth(:)), ...
        cos(elevation(:)).*sin(azimuth(:)),sin(elevation(:))]);
end
distance=clipped(:);
cloud=distance.*rays;
cloud(distance<=0 | distance>=7,:)=NaN;
end
