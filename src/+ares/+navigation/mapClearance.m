function [distance,normal] = mapClearance(position,map)
%MAPCLEARANCE Signed center-to-box distances and outward unit normals.
p=reshape(double(position),1,3);
center=(map.BoxMin+map.BoxMax)/2; half=(map.BoxMax-map.BoxMin)/2;
q=abs(p-center)-half;
distance=vecnorm(max(q,0),2,2)+min(max(q,[],2),0);
closest=min(max(p,map.BoxMin),map.BoxMax);
delta=p-closest; normal=delta./max(vecnorm(delta,2,2),eps);
inside=find(distance<=0);
for k=inside'
    [~,axis]=max(q(k,:)); normal(k,:)=0;
    signum=sign(p(axis)-center(k,axis)); if signum==0, signum=1; end
    normal(k,axis)=signum;
end
end
