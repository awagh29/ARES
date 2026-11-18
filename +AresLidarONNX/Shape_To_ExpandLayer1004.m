classdef Shape_To_ExpandLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end


    methods(Static, Hidden)
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'AresLidarONNX.coder.Shape_To_ExpandLayer1004';
        end
    end


    methods
        function this = Shape_To_ExpandLayer1004(name)
            this.Name = name;
            this.OutputNames = {'x_reconstruction_139'};
        end

        function [x_reconstruction_139] = predict(this, x_reconstruction_146)
            if isdlarray(x_reconstruction_146)
                x_reconstruction_146 = stripdims(x_reconstruction_146);
            end
            x_reconstruction_146NumDims = 3;
            x_reconstruction_146 = AresLidarONNX.ops.permuteInputVar(x_reconstruction_146, [3 2 1], 3);

            [x_reconstruction_139, x_reconstruction_139NumDims] = Shape_To_ExpandGraph1010(this, x_reconstruction_146, x_reconstruction_146NumDims, false);
            x_reconstruction_139 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction_139, [3 2 1], 3);

            x_reconstruction_139 = dlarray(single(x_reconstruction_139), 'CB');
        end

        function [x_reconstruction_139] = forward(this, x_reconstruction_146)
            if isdlarray(x_reconstruction_146)
                x_reconstruction_146 = stripdims(x_reconstruction_146);
            end
            x_reconstruction_146NumDims = 3;
            x_reconstruction_146 = AresLidarONNX.ops.permuteInputVar(x_reconstruction_146, [3 2 1], 3);

            [x_reconstruction_139, x_reconstruction_139NumDims] = Shape_To_ExpandGraph1010(this, x_reconstruction_146, x_reconstruction_146NumDims, true);
            x_reconstruction_139 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction_139, [3 2 1], 3);

            x_reconstruction_139 = dlarray(single(x_reconstruction_139), 'CB');
        end

        function [x_reconstruction_139, x_reconstruction_139NumDims1011] = Shape_To_ExpandGraph1010(this, x_reconstruction_146, x_reconstruction_146NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction_143, x_reconstruction_143NumDims] = AresLidarONNX.ops.onnxShape(x_reconstruction_146, x_reconstruction_146NumDims, 0, x_reconstruction_146NumDims+1);

            % Gather:
            [x_reconstruction_142, x_reconstruction_142NumDims] = AresLidarONNX.ops.onnxGather(x_reconstruction_143, this.Vars.x_reconstruction_135, 0, x_reconstruction_143NumDims, this.NumDims.x_reconstruction_135);

            % Unsqueeze:
            [shape, onnx__Concat_208NumDims] = AresLidarONNX.ops.prepareUnsqueezeArgs(x_reconstruction_142, this.Vars.onnx__Unsqueeze_207, x_reconstruction_142NumDims);
            onnx__Concat_208 = reshape(x_reconstruction_142, shape);

            % Concat:
            [x_reconstruction_134, x_reconstruction_134NumDims] = AresLidarONNX.ops.onnxConcat(0, {this.Vars.onnx__Concat_346, onnx__Concat_208, this.Vars.x_reconstruction_136}, [this.NumDims.onnx__Concat_346, onnx__Concat_208NumDims, this.NumDims.x_reconstruction_136]);

            % Expand:
            [shape, x_reconstruction_139NumDims] = AresLidarONNX.ops.prepareExpandArgs(this.NumDims.x_reconstruction_138, x_reconstruction_134);
            x_reconstruction_139 = this.Vars.x_reconstruction_138 + zeros(shape);

            % Set graph output arguments
            x_reconstruction_139NumDims1011 = x_reconstruction_139NumDims;

        end

    end

end