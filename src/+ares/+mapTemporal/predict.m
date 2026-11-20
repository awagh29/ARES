function [ranges,diagnostics] = predict(model,observed,valid,available,buffer)
%PREDICT Native FP32 known-map sparse CNN + measured-history GRU inference.
available=logical(available); valid=logical(valid)&available;
observed=single(observed); observed(~available)=0;
assert(isequal(size(observed),[15 716]) && all(isfinite(observed),'all'), ...
    'ARES:TemporalInput','Expected finite observed 15x716 ranges.');
hit=reshape(single(valid),1,15,716); support=reshape(single(available),1,15,716);
inputs=cat(1,reshape(min(max(1-observed/model.maxRange,0),1),1,15,716).*hit,hit,support);
h=zeros(double(model.historyHidden),1,'single');
boxes=single(model.obstacles); center=mean(boxes(:,1:3),1); half=boxes(:,4:6)/2;
H=double(model.historySteps);
for t=1:H
    if ~buffer.Present(t), continue; end
    state=buffer.States(t,:); rpy=state(7:9); R=rotation(rpy);
    delta=boxes(:,1:3)-state(1:3);
    closest=delta+min(max(-delta,-half),half);
    geometry=[R'*delta';boxes(:,4:6)';R'*closest']/model.positionScale;
    obstacle=silu(model.obstacleFirst.weight*geometry+model.obstacleFirst.bias);
    obstacle=silu(model.obstacleSecond.weight*obstacle+model.obstacleSecond.bias);
    obstacle=max(obstacle,[],2);
    encoded=[(state(1:3)-center)'/model.positionScale;state(4:6)'/model.velocityScale; ...
        reshape([sin(rpy);cos(rpy)],[],1);R'*(buffer.Goals(t,:)-state(1:3))'/model.positionScale; ...
        single(t-H)*single(model.samplePeriodS);log1p(max(buffer.Ages(t),0));single(1)];
    x=max(convolution(buffer.Summaries(:,:,:,t),model.historyConv),0);
    pooled=zeros(size(x,1),12,'single');
    for bin=1:12
        first=floor((bin-1)*90/12)+1; last=ceil(bin*90/12);
        pooled(:,bin)=mean(x(:,:,first:last),[2 3]);
    end
    temporal=max(model.historyLinear.weight*reshape(pooled',[],1)+model.historyLinear.bias,0);
    input=[temporal;encoded;obstacle];
    a=model.gruInputWeight*input+model.gruInputBias;
    b=model.gruHiddenWeight*h+model.gruHiddenBias;
    n=numel(h); r=sigmoid(a(1:n)+b(1:n)); z=sigmoid(a(n+(1:n))+b(n+(1:n)));
    candidate=tanh(a(2*n+(1:n))+r.*b(2*n+(1:n)));
    h=(1-z).*candidate+z.*h;
end
condition=[h;encoded;obstacle];
x=inputs(1:2,:,:); mask=support; skips=cell(1,numel(model.encoder)); masks=skips;
for k=1:numel(model.encoder)
    block=model.encoder{k}; [x,mask]=sparseConv(x,mask,block.first);
    [x,mask]=sparseConv(x,mask,block.second); skips{k}=x; masks{k}=mask;
end
film=model.filmSecond.weight*max(model.filmFirst.weight*condition+model.filmFirst.bias,0)+model.filmSecond.bias;
c=size(x,1); x=x.*(1+reshape(film(1:c),c,1,1))+reshape(film(c+1:end),c,1,1);
for k=1:numel(model.decoder)
    skip=numel(skips)-k; x=cat(1,x,mask); x=x(:,:,repelem(1:size(x,3),2));
    x=cat(1,x,skips{skip},masks{skip}); block=model.decoder{k};
    x=max(convolution(x,block.first),0); x=max(convolution(x,block.second),0); mask=masks{skip};
end
[prior,priorHit]=raycast(model,buffer.States(end,:));
mapInput=cat(1,reshape((1-prior/model.maxRange).*single(priorHit),1,15,716),reshape(single(priorHit),1,15,716));
mapFeatures=max(convolution(mapInput,model.mapFirst),0);
mapFeatures=max(convolution(mapFeatures,model.mapSecond),0);
features=cat(1,x,mask,inputs,mapFeatures,mapInput);
correction=model.residualLimit*tanh(reshape(convolution(features,model.head),15,716));
ranges=min(max(prior+correction,0),model.maxRange); ranges(available)=observed(available);
diagnostics=struct('Prior',prior,'Correction',correction,'Conditioning',condition);
end

function R=rotation(rpy)
r=rpy(1); p=rpy(2); y=rpy(3); sr=sin(r);cr=cos(r);sp=sin(p);cp=cos(p);sy=sin(y);cy=cos(y);
R=[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr; ...
    sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr;-sp,cp*sr,cp*cr];
end
function y=sigmoid(x), y=1./(1+exp(-x)); end
function y=silu(x), y=x.*sigmoid(x); end
function [ranges,hits]=raycast(m,state)
R=rotation(state(7:9)); origin=state(1:3)+(R*m.mountingTranslation(:))';
direction=m.raysBody*R'; nearest=Inf(10740,1,'single');
for box=1:size(m.obstacles,1)
    lo=m.obstacles(box,1:3)-m.obstacles(box,4:6)/2;
    hi=m.obstacles(box,1:3)+m.obstacles(box,4:6)/2;
    near=-Inf(10740,1,'single'); far=Inf(10740,1,'single'); outside=false(10740,1);
    for axis=1:3
        parallel=abs(direction(:,axis))<single(1e-8); divisor=direction(:,axis); divisor(parallel)=1;
        first=(lo(axis)-origin(axis))./divisor; second=(hi(axis)-origin(axis))./divisor;
        lower=min(first,second); upper=max(first,second); lower(parallel)=-Inf; upper(parallel)=Inf;
        outside=outside | (parallel & (origin(axis)<lo(axis) || origin(axis)>hi(axis)));
        near=max(near,lower); far=min(far,upper);
    end
    candidate=near; candidate(near<0)=far(near<0);
    candidate(far<max(near,0) | outside | candidate<=0)=Inf;
    nearest=min(nearest,candidate);
end
hits=isfinite(nearest)&nearest>0&nearest<=m.maxRange;
nearest(~hits)=m.maxRange; ranges=reshape(nearest,15,716); hits=reshape(hits,15,716);
end
function [y,maskOut]=sparseConv(x,mask,layer)
supportLayer=layer; supportLayer.weight=ones(1,prod(double(layer.kernel)),'single'); supportLayer.bias=single(0);
support=convolution(mask,supportLayer); maskOut=single(support>0);
noBias=layer; noBias.bias=zeros(size(layer.bias),'single');
y=max(convolution(x.*mask,noBias)./max(support,single(1e-6))+layer.bias,0).*maskOut;
end
function y=convolution(x,layer)
[channels,height,width]=size(x); kernel=double(layer.kernel); stride=double(layer.stride);
ke=kernel(1); ka=kernel(2); pe=floor(ke/2); pa=floor(ka/2);
padded=zeros(channels,height+2*pe,width+2*pa,'single');
if layer.circularAzimuth
    padded(:,pe+(1:height),:)=x(:,:,mod((-pa):(width+pa-1),width)+1);
else
    padded(:,pe+(1:height),pa+(1:width))=x;
end
ho=floor((height+2*pe-ke)/stride(1))+1; wo=floor((width+2*pa-ka)/stride(2))+1;
patches=zeros(channels*ke*ka,ho*wo,'single');
for azimuth=1:ka
    for elevation=1:ke
        rows=(azimuth-1)*ke*channels+(elevation-1)*channels+(1:channels);
        patch=padded(:,elevation:stride(1):elevation+(ho-1)*stride(1),azimuth:stride(2):azimuth+(wo-1)*stride(2));
        patches(rows,:)=reshape(patch,channels,[]);
    end
end
y=reshape(layer.weight*patches+layer.bias,[],ho,wo);
end
