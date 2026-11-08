function [physical, normalized, details] = predictSaits(model, rawFeatures, observedMask)
%PREDICTSAITS Native FP32 inference for the exported SAITS endpoint model.
%#codegen
% rawFeatures contains chronological physical-unit rows, ending at the
% prediction time. Supply NaN for missing inputs, or pass observedMask with
% finite placeholders. No Deep Learning Toolbox or Python runtime is used.
% Outputs are row vectors. Physical outputs are deliberately not clipped.
%
% The final second-stage attention layer evaluates only the endpoint query
% and its feed-forward branch. Earlier layers retain every historical row.

assert(isnumeric(rawFeatures) && isreal(rawFeatures) && ismatrix(rawFeatures), ...
    'ARES:InvalidSaitsFeatures', 'SAITS features must be a real numeric matrix.');
assert(size(rawFeatures, 1) == double(model.nSteps) && ...
    size(rawFeatures, 2) == 31 && double(model.nFeatures) == 31 && ...
    double(model.nTargets) == 14 && size(rawFeatures, 1) >= 2, ...
    'ARES:InvalidSaitsShape', 'SAITS requires the exported history length and 31 input features.');

if nargin < 3
    observedMask = isfinite(rawFeatures);
end
assert(isequal(size(observedMask), size(rawFeatures)) && ...
    all(observedMask(:) == 0 | observedMask(:) == 1), ...
    'ARES:InvalidSaitsMask', 'The observation mask must be binary and match the input shape.');
assert(all(isfinite(rawFeatures(:)) | ~logical(observedMask(:))), ...
    'ARES:NonfiniteSaitsObservation', 'Observed SAITS features must be finite.');
assert(numel(model.contextMean) == 17 && numel(model.contextStd) == 17 && ...
    all(isfinite(model.contextMean(:))) && all(isfinite(model.contextStd(:))) && ...
    all(model.contextStd(:) > 0), ...
    'ARES:InvalidSaitsTransform', 'SAITS context statistics must be finite with positive scales.');
assert(numel(model.firstBlock) >= 1 && ...
    numel(model.firstBlock) == numel(model.secondBlock), ...
    'ARES:InvalidSaitsLayers', 'Both SAITS stages need the exported attention layers.');

% Match FeatureTransform exactly: first cast raw values to FP32. NumPy then
% subtracts/divides FP64 context statistics before assignment back to FP32.
filled = single(rawFeatures);
filled(:, 1:7) = filled(:, 1:7) ./ single(model.distanceScale);
filled(:, 15:31) = single((double(filled(:, 15:31)) - ...
    double(reshape(model.contextMean, 1, 17))) ./ ...
    double(reshape(model.contextStd, 1, 17)));
mask = single(observedMask);
filled(~logical(observedMask)) = single(0);

hidden = linear([filled, mask], model.embeddingFirst) + single(model.positionEncoding);
for index = 1:numel(model.firstBlock)
    hidden = fullLayer(hidden, model.firstBlock{index}, ...
        double(model.nHeads), double(model.headDim));
end
first = linear(hidden, model.reduceFirst);
completed = mask .* filled + (single(1) - mask) .* first;
hidden = linear([completed, mask], model.embeddingSecond) + single(model.positionEncoding);

for index = 1:numel(model.secondBlock)
    if index == numel(model.secondBlock)
        [endpoint, attentionEndpoint] = endpointLayer(hidden, model.secondBlock{index}, ...
            double(model.nHeads), double(model.headDim));
    else
        hidden = fullLayer(hidden, model.secondBlock{index}, ...
            double(model.nHeads), double(model.headDim));
    end
end
second = linear(max(linear(endpoint, model.reduceSecondBeta), single(0)), ...
    model.reduceSecondGamma);
logits = linear([mask(end, :), attentionEndpoint], model.combinationGate);
mixing = single(1) ./ (single(1) + exp(-logits(1, 1:14)));
firstEndpoint = first(end, 1:14);
secondEndpoint = second(1, 1:14);
normalized = (single(1) - mixing) .* secondEndpoint + mixing .* firstEndpoint;
physical = normalized .* single(reshape(model.targetScale, 1, 14));
details.stage1Normalized = firstEndpoint;
details.stage2Normalized = secondEndpoint;
details.attentionEndpoint = attentionEndpoint;
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
