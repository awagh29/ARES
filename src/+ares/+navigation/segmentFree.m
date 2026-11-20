function free = segmentFree(first,last,map,padding,bounds)
%SEGMENTFREE Exact segment/slab test against conservatively expanded boxes.
first=reshape(first,1,3); last=reshape(last,1,3);
if any(first<bounds(:,1)' | first>bounds(:,2)' | last<bounds(:,1)' | last>bounds(:,2)')
    free=false; return;
end
lo=map.BoxMin-padding; hi=map.BoxMax+padding; direction=last-first;
tmin=zeros(size(lo,1),1); tmax=ones(size(lo,1),1);
for axis=1:3
    if abs(direction(axis))<1e-12
        outside=first(axis)<lo(:,axis) | first(axis)>hi(:,axis);
        tmin(outside)=Inf;
    else
        a=(lo(:,axis)-first(axis))/direction(axis);
        b=(hi(:,axis)-first(axis))/direction(axis);
        tmin=max(tmin,min(a,b)); tmax=min(tmax,max(a,b));
    end
end
free=~any(tmin<=tmax);
end
