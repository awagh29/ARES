function offsets = cloudOffsets(points,eulerZYX)
%CLOUDOFFSETS Selected sensor cloud -> world-oriented offsets from UAV centre.
% Uses only the measured/reconstructed cloud, never the nominal scoring scan.
xyz=double(points);
ranges=vecnorm(xyz,2,2);
keep=all(isfinite(xyz),2) & ranges>0 & ranges<7;
body=xyz(keep,:)*diag([1 -1 -1])+[0 0 -.4];
offsets=body*eul2rotm(reshape(double(eulerZYX),1,3),'ZYX')';
end
