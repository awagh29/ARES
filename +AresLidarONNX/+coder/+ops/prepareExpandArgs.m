function [shape, numDimsY] = prepareExpandArgs(NumDimX, ONNXShape_)
% Prepares arguments for implementing the ONNX Expand operator
%#codegen

%   Copyright 2024 The MathWorks, Inc.    

% Broadcast X to ONNXShape. The shape of X must be compatible with ONNXShape.
ONNXShape = AresLidarONNX.coder.ops.extractIfDlarray(ONNXShape_);

shape_ = fliplr(ONNXShape(:)');
if numel(shape_) < 2
    shape = [shape_ ones(1, 2 - numel(shape_))];
else
    shape = shape_;
end
if numel(ONNXShape) > NumDimX
    numDimsY = numel(ONNXShape);
else
    numDimsY = NumDimX;
end
end
