function [Y, numDimsY] = onnxConcat(ONNXAxis_, XCell_, numDimsXArray_)
%#codegen
% Copyright 2024-2026 The MathWorks, Inc.

ONNXAxis      = AresLidarONNX.coder.ops.extractIfDlarray(ONNXAxis_);
numDimsXArray = AresLidarONNX.coder.ops.extractIfDlarray(numDimsXArray_);
numDimsY      = numDimsXArray(1);

if isempty(XCell_)
    Y = dlarray([]);
else
    numTensors = coder.const(numel(XCell_));
    XCell = cell(1, numTensors);
    coder.unroll();
    for i = 1:numTensors
        if isempty(XCell_{i})
            XCell{i} = [];
        else
            XCell{i} = XCell_{i};
        end
    end

    if ONNXAxis < 0
        ONNXAxis = ONNXAxis + numDimsY;
    end
    DLTAxis = coder.const(numDimsY - ONNXAxis);

    % Avoid cat() during isAmbiguousTypes pass ---
    % cat() wraps its result in coder.ignoreConst during the
    % isAmbiguousTypes pass, which breaks callers that need a
    % constant-foldable result.  Route axis==1 and axis==2 through
    % vertcat/horzcat (which are not affected), and for the general
    % case synthesize a size-correct placeholder that is NOT wrapped
    % in ignoreConst so callers can constant-fold it.
    if coder.const(DLTAxis == 1)
        Y = vertcat(XCell{:});
    elseif coder.const(DLTAxis == 2)
        Y = horzcat(XCell{:});
    elseif coder.internal.isAmbiguousTypes
        % Build a zeros placeholder with the correct size so that
        % type/size propagation succeeds without ignoreConst poisoning
        % the result.  The actual element values are irrelevant here;
        % only the type and size matter for this pass.
        ysize = size(XCell{1}, 1:numDimsY);
        for k = coder.unroll(2:numTensors)
            ysize(DLTAxis) = ysize(DLTAxis) + size(XCell{k}, DLTAxis);
        end
        Y = zeros(ysize, 'like', XCell{1});   % no coder.ignoreConst
    else
        Y = cat(DLTAxis, XCell{:});
    end
end
end