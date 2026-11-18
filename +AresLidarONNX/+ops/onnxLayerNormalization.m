function [Y, numDimsY] = onnxLayerNormalization(X, scale, offset, axis, epsilon, numDimsX)
%onnxLayerNormalization Normalizes mini-batches of data over the normalized shape.

%Copyright 2024 The MathWorks, Inc.

% axis is in forward ONNX format. Convert axis to reverse ONNX format. 
if (axis<0)
    axis = -axis;
else
    axis = numDimsX - axis; 
end

epsilon = double(epsilon);
% Epsilon must be non-negative.
if(epsilon <= 0)
    warning(message("nnet_cnn_onnx:onnx:BadEpsilon"));
    epsilon = 1e-5;
end

dimension = 1: axis;

% stage 1: standarization
XMean = mean(X, dimension);
D = X - XMean;
DD = D.^2; 
XVar = mean(DD, dimension);
XVarEps = XVar + epsilon;
StdDev = sqrt(XVarEps);
InvStdDev  = 1 ./ StdDev;
Normalized = D .* InvStdDev;

% stage 2
NormalizedScaled = Normalized .* scale;
if ~isempty(offset)
    Y = NormalizedScaled + offset;
else
    Y = NormalizedScaled;
end
numDimsY = numDimsX;
end