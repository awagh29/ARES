function map = sceneMap(obstaclePositionsEnu,width,height)
%SCENEMAP Convert the existing ENU polygon meshes into actual NED AABBs.
% initializeSingleUav names its loop variables north/east, but addMesh uses ENU.
centers=unique(double(obstaclePositionsEnu(:,[2 1])),'rows','stable');
map.BoxMin=[centers-width/2, -height*ones(size(centers,1),1)];
map.BoxMax=[centers+width/2, zeros(size(centers,1),1)];
map.Frame="NED";
map.Source="existing initializeSingleUav polygon meshes; ENU converted to NED";
map.SensorMountBodyM=[0 0 -.4];
map.SensorToBodyRotation=diag([1 -1 -1]);
map.MaxRangeM=7;
[az,el]=meshgrid(linspace(-179,179,716)*pi/180,linspace(-15,15,15)*pi/180);
map.SensorDirections=[cos(el(:)).*cos(az(:)),cos(el(:)).*sin(az(:)),sin(el(:))];
end
