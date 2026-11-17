function [ranges, normalized, details] = predictDenseSaits(model,rawFeatures,observedMask)
%PREDICTDENSESAITS Native FP32 full-scan reconstruction; no Python/QP.
assert(string(model.schemaVersion)=="ares-dense-saits-single-episode-mat-v1", ...
    "ARES:WrongDenseSaitsSchema","Wrong dense SAITS artifact.");
assert(isequal(size(rawFeatures),[double(model.nSteps) 10752]) && ...
    isequal(size(observedMask),size(rawFeatures)),"ARES:DenseSaitsInputShape","Expected window x 10752 inputs/masks.");
assert(all(observedMask(:)==0|observedMask(:)==1) && all(isfinite(rawFeatures(logical(observedMask)))), ...
    "ARES:DenseSaitsInvalidInput","Observed values must be finite and mask binary.");
assert(numel(model.contextMean)==12 && numel(model.contextStd)==12 && ...
    all(isfinite(model.contextMean)) && all(isfinite(model.contextStd)) && all(model.contextStd>0), ...
    "ARES:DenseSaitsNormalizer","Expected twelve finite training-state statistics.");
mask=single(observedMask); filled=single(rawFeatures); filled(~logical(observedMask))=0;
filled(:,1:10740)=filled(:,1:10740)./single(7);
filled(:,10741:10752)=single((double(filled(:,10741:10752))-double(model.contextMean))./double(model.contextStd));
filled(~logical(observedMask))=0;
hidden=linear([filled mask],model.embeddingFirst)+single(model.positionEncoding);
for i=1:numel(model.firstBlock)
    hidden=fullLayer(hidden,model.firstBlock{i},double(model.nHeads),double(model.headDim));
end
first=linear(hidden,model.reduceFirst);
completed=mask.*filled+(single(1)-mask).*first;
hidden=linear([completed mask],model.embeddingSecond)+single(model.positionEncoding);
for i=1:numel(model.secondBlock)
    if i==numel(model.secondBlock)
        [endpoint,attention]=endpointLayer(hidden,model.secondBlock{i},double(model.nHeads),double(model.headDim));
    else
        hidden=fullLayer(hidden,model.secondBlock{i},double(model.nHeads),double(model.headDim));
    end
end
second=linear(max(linear(endpoint,model.reduceSecondBeta),single(0)),model.reduceSecondGamma);
logits=linear(max(linear([mask(end,:) attention],model.gateFirst),single(0)),model.gateSecond);
mixing=single(1)./(single(1)+exp(-logits));
firstEndpoint=first(end,1:10740);
normalized=(single(1)-mixing).*second+mixing.*firstEndpoint;
ranges=reshape(normalized.*single(7),15,716);
details.stage1Normalized=firstEndpoint; details.stage2Normalized=second;
end

function output = linear(input, parameters)
output = input * single(parameters.W) + single(parameters.b);
end

function normalized = layerNorm(input, parameters)
centered = input - mean(input, 2);
variance = mean(centered .* centered, 2);
normalized = (centered ./ sqrt(variance + single(parameters.epsilon))) .* ...
    single(parameters.scale) + single(parameters.bias);
end

function hidden = fullLayer(hidden, layer, heads, headDimension)
steps = size(hidden, 1);
normalized = layerNorm(hidden, layer.attentionNorm);
query = normalized * single(layer.queryW);
key = normalized * single(layer.keyW);
value = normalized * single(layer.valueW);
attended = zeros(size(hidden), 'single');
scale = single(1 / sqrt(headDimension));
for head = 1:heads
    columns = (head - 1) * headDimension + (1:headDimension);
    scores = (query(:, columns) * key(:, columns).') .* scale;
    for row = 1:steps
        scores(row, row) = single(-inf);
    end
    weights = rowSoftmax(scores);
    attended(:, columns) = weights * value(:, columns);
end
hidden = hidden + attended * single(layer.outputW);
feedForward = linear(max(linear(layerNorm(hidden, layer.ffNorm), ...
    layer.ffFirst), single(0)), layer.ffSecond);
hidden = hidden + feedForward;
end

function [endpoint, averageAttention] = endpointLayer(hidden, layer, heads, headDimension)
steps = size(hidden, 1);
normalized = layerNorm(hidden, layer.attentionNorm);
query = normalized(end, :) * single(layer.queryW);
key = normalized * single(layer.keyW);
value = normalized * single(layer.valueW);
attended = zeros(1, size(hidden, 2), 'single');
averageAttention = zeros(1, steps, 'single');
scale = single(1 / sqrt(headDimension));
for head = 1:heads
    columns = (head - 1) * headDimension + (1:headDimension);
    scores = (query(:, columns) * key(:, columns).') .* scale;
    scores(steps) = single(-inf);
    weights = rowSoftmax(scores);
    attended(:, columns) = weights * value(:, columns);
    averageAttention = averageAttention + weights ./ single(heads);
end
endpoint = hidden(end, :) + attended * single(layer.outputW);
feedForward = linear(max(linear(layerNorm(endpoint, layer.ffNorm), ...
    layer.ffFirst), single(0)), layer.ffSecond);
endpoint = endpoint + feedForward;
end

function probability = rowSoftmax(scores)
exponent = exp(scores - max(scores, [], 2));
probability = exponent ./ sum(exponent, 2);
end
