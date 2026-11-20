function ranges = predictSparseReconstruction(model,ranges,valid,available,rawState)
%PREDICTSPARSERECONSTRUCTION Native FP32 sparse CNN/FiLM notebook inference.
% Arrays use [channel,elevation,azimuth]; beam flattening remains MATLAB order.
assert(isequal(size(ranges),[15 716]) && isequal(size(valid),[15 716]) && ...
    isequal(size(available),[15 716]),"ARES:SparseInputShape","Expected 15x716 input grids.");
rawState=reshape(single(rawState),1,9);
assert(all(isfinite(rawState)),"ARES:SparseState","Nonfinite state.");
available=logical(available); valid=logical(valid)&available;
ranges=single(ranges); ranges(~available)=0;
assert(all(isfinite(ranges),"all"),"ARES:SparseRanges","Nonfinite measured ranges.");
hit=reshape(single(valid),1,15,716);
support=reshape(single(available),1,15,716);
nearness=min(max(1-ranges/single(model.maxRange),0),1);
inputs=cat(1,reshape(nearness,1,15,716).*hit,hit,support);
x=inputs(1:2,:,:); mask=support;
skips=cell(1,numel(model.encoder)); masks=skips;
for k=1:numel(model.encoder)
    block=model.encoder{k};
    [x,mask]=sparseConv(x,mask,block.first);
    [x,mask]=sparseConv(x,mask,block.second);
    skips{k}=x; masks{k}=mask;
end
rpy=rawState(7:9);
angles=reshape([sin(rpy);cos(rpy)],1,6);
if model.usePosition, state=[rawState(1:6),angles]; else, state=[rawState(4:6),angles]; end
film=model.filmSecond.weight*max(model.filmFirst.weight*state.'+model.filmFirst.bias,0)+model.filmSecond.bias;
c=size(x,1); gamma=reshape(film(1:c),c,1,1); beta=reshape(film(c+1:end),c,1,1);
x=x.*(1+gamma)+beta;
for k=1:numel(model.decoder)
    skip=numel(skips)-k;
    x=cat(1,x,mask); x=x(:,:,repelem(1:size(x,3),2));
    x=cat(1,x,skips{skip},masks{skip}); block=model.decoder{k};
    x=max(ringConv(x,block.first),0);
    x=max(ringConv(x,block.second),0); mask=masks{skip};
end
features=cat(1,x,mask,inputs);
logits=ringConv(features,model.head);
ranges=reshape(single(model.maxRange)./(1+exp(-logits)),15,716);
end

function [y,maskOut]=sparseConv(x,mask,layer)
% Sum support and max-pool support agree for this binary availability mask.
kernel=double(layer.kernel); stride=double(layer.stride);
supportLayer=struct("kernel",kernel,"stride",stride, ...
    "weight",ones(1,prod(kernel),"single"),"bias",single(0));
support=ringConv(mask,supportLayer);
maskOut=single(support>0);
noBias=layer; noBias.bias=zeros(size(layer.bias),"single");
y=ringConv(x.*mask,noBias)./max(support,single(1e-6))+layer.bias;
y=max(y,0).*maskOut;
end

function y=ringConv(x,layer)
% im2col without Image Processing/Deep Learning Toolbox; matrix product uses BLAS.
[channels,height,width]=size(x);
kernel=double(layer.kernel); stride=double(layer.stride);
ke=kernel(1); ka=kernel(2); pe=floor(ke/2); pa=floor(ka/2);
padded=zeros(channels,height+2*pe,width+2*pa,"single");
padded(:,pe+(1:height),:)=x(:,:,mod((-pa):(width+pa-1),width)+1);
ho=floor((height+2*pe-ke)/stride(1))+1;
wo=floor((width+2*pa-ka)/stride(2))+1;
patches=zeros(channels*ke*ka,ho*wo,"single");
for azimuth=1:ka
    for elevation=1:ke
        rows=(azimuth-1)*ke*channels+(elevation-1)*channels+(1:channels);
        patch=padded(:,elevation:stride(1):elevation+(ho-1)*stride(1), ...
            azimuth:stride(2):azimuth+(wo-1)*stride(2));
        patches(rows,:)=reshape(patch,channels,[]);
    end
end
y=reshape(layer.weight*patches+layer.bias,[],ho,wo);
end
