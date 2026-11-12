function [ranges, encoded] = predictCnnMlp(model, rawState)
%PREDICTCNNMLP Native FP32 inference; no Python or neural-network toolbox.
% Input is [N E D vN vE vD roll pitch yaw], angles in radians.
assert(string(model.schemaVersion)=="ares-cnn-mlp-single-episode-mat-v1", ...
    "ARES:InvalidCnnSchema","Unsupported CNN artifact.");
raw = reshape(single(rawState),1,[]);
assert(numel(raw)==9 && all(isfinite(raw)),"ARES:InvalidCnnInput","Expected nine finite raw state values.");
angles=raw(7:9);
encoded=[raw(1:6),reshape([sin(angles);cos(angles)],1,6)];
features=encoded.';
for k=1:numel(model.encoder)
    layer=model.encoder{k};
    features=activate(single(layer.W)*features+single(layer.b(:)),layer.activation);
end
features=reshape(features,double(model.latentLength),double(model.latentChannels)).';
for k=1:numel(model.decoder)
    layer=model.decoder{k};
    features=upsample(features,double(layer.length),logical(model.alignCorners));
    W=single(layer.W); width=size(W,3); padding=(width-1)/2;
    padded=zeros(size(features,1),size(features,2)+2*padding,"single");
    padded(:,padding+(1:size(features,2)))=features;
    output=repmat(single(layer.b(:)),1,size(features,2));
    for offset=1:width
        output=output+W(:,:,offset)*padded(:,offset:offset+size(features,2)-1);
    end
    features=activate(output,layer.activation);
end
assert(isequal(size(features),[15 716]),"ARES:InvalidCnnOutput","Expected a 15x716 scan.");
ranges=features; % Physical clipping belongs to the sensor/controller adapter.
end

function out=upsample(values,n,alignCorners)
old=size(values,2);
if old==n, out=values; return; end
if alignCorners && n>1
    source=single(0:n-1)*single((old-1)/(n-1));
else
    source=(single(0:n-1)+single(.5))*single(old/n)-single(.5);
end
source=max(source,single(0));
left=min(floor(source),old-1);
right=min(left+1,old-1);
weight=source-left;
out=values(:,double(left)+1).*(single(1)-weight)+values(:,double(right)+1).*weight;
end

function value=activate(value,name)
switch string(name)
    case "gelu"
        value=single(.5)*value.*(single(1)+erf(value/sqrt(single(2))));
    case "relu"
        value=max(value,single(0));
    case "silu"
        value=value./(single(1)+exp(-value));
    case "softplus"
        value=max(value,single(0))+log1p(exp(-abs(value)));
    case "identity"
    otherwise
        error("ARES:UnknownCnnActivation","Unsupported activation: %s",string(name));
end
end
