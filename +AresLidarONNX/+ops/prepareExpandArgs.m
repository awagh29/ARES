function [shape, numDimsY] = prepareExpandArgs(NumDimX, ONNXShape)
% Prepares arguments for implementing the ONNX Expand operator

%   Copyright 2020-2025 The MathWorks, Inc.    

% Broadcast X to ONNXShape. The shape of X must be compatible with ONNXShape.
ONNXShape = extractdata(ONNXShape);
shape = fliplr(ONNXShape(:)');
if numel(shape) < 2
    shape = [shape ones(1, 2-numel(shape))];
end
if numel(ONNXShape) > NumDimX
    numDimsY = numel(ONNXShape);
else
    numDimsY = NumDimX;
end
end
