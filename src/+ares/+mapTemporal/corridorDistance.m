function distance = corridorDistance(offsets,direction,radius,horizon)
%CORRIDORDISTANCE First point-obstacle contact along a swept spherical body.
% Sparse scan surfaces are an approximation, not a free-space certificate.
direction=reshape(double(direction),3,1);
if norm(direction)<1e-9, distance=horizon; return; end
direction=direction/norm(direction);
along=offsets*direction;
lateral2=max(sum(offsets.^2,2)-along.^2,0);
reach=sqrt(max(radius^2-lateral2,0));
% Once inside the engineering margin, permit motion away from a return.
% Counting a rearward overlap as a new forward contact traps a stopped UAV.
hit=lateral2<=radius^2 & along>0;
entry=max(along(hit)-reach(hit),0);
distance=min([horizon;entry]);
end
