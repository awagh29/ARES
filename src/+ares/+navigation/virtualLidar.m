function [cloud,ranges,hits] = virtualLidar(map,position,eulerZYX)
%VIRTUALLIDAR Deterministic 15x716 physical-map ray casts, sensor-FLU output.
rotation=eul2rotm(reshape(double(eulerZYX),1,3),'ZYX');
origin=reshape(double(position),1,3)+(rotation*map.SensorMountBodyM')';
direction=map.SensorDirections*map.SensorToBodyRotation'*rotation';
nearest=Inf(size(direction,1),1);
for k=1:size(map.BoxMin,1)
    near=-Inf(size(nearest)); far=Inf(size(nearest));
    for axis=1:3
        parallel=abs(direction(:,axis))<1e-12;
        divisor=direction(:,axis); divisor(parallel)=1;
        a=(map.BoxMin(k,axis)-origin(axis))./divisor;
        b=(map.BoxMax(k,axis)-origin(axis))./divisor;
        lower=min(a,b); upper=max(a,b);
        lower(parallel)=-Inf; upper(parallel)=Inf;
        if origin(axis)<map.BoxMin(k,axis) || origin(axis)>map.BoxMax(k,axis)
            lower(parallel)=Inf;
        end
        near=max(near,lower); far=min(far,upper);
    end
    valid=far>=max(near,0); candidate=near;
    candidate(near<0)=far(near<0);
    candidate(~valid | candidate<=0)=Inf;
    nearest=min(nearest,candidate);
end
hits=isfinite(nearest) & nearest<=map.MaxRangeM;
ranges=map.MaxRangeM*ones(size(nearest)); ranges(hits)=nearest(hits);
cloud=NaN(numel(nearest),3,'single');
cloud(hits,:)=single(map.SensorDirections(hits,:).*ranges(hits));
end
