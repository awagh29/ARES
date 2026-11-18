classdef Shape_To_MulLayer1002 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        x_reconstruction__89
        x_reconstruction__88
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
            name = 'AresLidarONNX.coder.Shape_To_MulLayer1002';
        end
    end


    methods
        function this = Shape_To_MulLayer1002(name)
            this.Name = name;
            this.OutputNames = {'x_reconstruction_103'};
        end

        function [x_reconstruction_103] = predict(this, x_reconstruction__86)
            if isdlarray(x_reconstruction__86)
                x_reconstruction__86 = stripdims(x_reconstruction__86);
            end
            x_reconstruction__86NumDims = 4;
            x_reconstruction__86 = AresLidarONNX.ops.permuteInputVar(x_reconstruction__86, [4 3 1 2], 4);

            [x_reconstruction_103, x_reconstruction_103NumDims] = Shape_To_MulGraph1004(this, x_reconstruction__86, x_reconstruction__86NumDims, false);
            x_reconstruction_103 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction_103, [3 4 2 1], 4);

            x_reconstruction_103 = dlarray(single(x_reconstruction_103), 'SSCB');
        end

        function [x_reconstruction_103] = forward(this, x_reconstruction__86)
            if isdlarray(x_reconstruction__86)
                x_reconstruction__86 = stripdims(x_reconstruction__86);
            end
            x_reconstruction__86NumDims = 4;
            x_reconstruction__86 = AresLidarONNX.ops.permuteInputVar(x_reconstruction__86, [4 3 1 2], 4);

            [x_reconstruction_103, x_reconstruction_103NumDims] = Shape_To_MulGraph1004(this, x_reconstruction__86, x_reconstruction__86NumDims, true);
            x_reconstruction_103 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction_103, [3 4 2 1], 4);

            x_reconstruction_103 = dlarray(single(x_reconstruction_103), 'SSCB');
        end

        function [x_reconstruction_103, x_reconstruction_103NumDims1005] = Shape_To_MulGraph1004(this, x_reconstruction__86, x_reconstruction__86NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction__95, x_reconstruction__95NumDims] = AresLidarONNX.ops.onnxShape(x_reconstruction__86, x_reconstruction__86NumDims, 0, x_reconstruction__86NumDims+1);

            % Reshape:
            [shape, x_reconstruction__94NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction__86, this.Vars.x_reconstruction__90, x_reconstruction__86NumDims, 0);
            x_reconstruction__94 = reshape(x_reconstruction__86, shape{:});

            % InstanceNormalization:
            if x_reconstruction__94NumDims > 2
                dataFormat = [repmat('S',[1 x_reconstruction__94NumDims-2]),'CB'];
                x_reconstruction__91 = instancenorm(x_reconstruction__94, this.x_reconstruction__89, this.x_reconstruction__88, 'Epsilon', 0.000010, 'DataFormat', dataFormat);
            else
                x_reconstruction__91 = x_reconstruction__94;
            end
            x_reconstruction__91NumDims = x_reconstruction__94NumDims;

            % Reshape:
            [shape, x_reconstruction__93NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction__91, x_reconstruction__95, x_reconstruction__91NumDims, 0);
            x_reconstruction__93 = reshape(x_reconstruction__91, shape{:});

            % Mul:
            x_reconstruction__92 = x_reconstruction__93 .* this.Vars.onnx__Mul_318;
            x_reconstruction__92NumDims = max(x_reconstruction__93NumDims, this.NumDims.onnx__Mul_318);

            % Add:
            x_reconstruction__87 = x_reconstruction__92 + this.Vars.onnx__Add_319;
            x_reconstruction__87NumDims = max(x_reconstruction__92NumDims, this.NumDims.onnx__Add_319);

            % Div:
            x_reconstruction_100 = x_reconstruction__87 ./ this.Vars.x_reconstruction__99;
            x_reconstruction_100NumDims = max(x_reconstruction__87NumDims, this.NumDims.x_reconstruction__99);

            % Erf:
            x_reconstruction_101 = erf(x_reconstruction_100);
            x_reconstruction_101NumDims = x_reconstruction_100NumDims;

            % Add:
            x_reconstruction__96 = x_reconstruction_101 + this.Vars.x_reconstruction__97;
            x_reconstruction__96NumDims = max(x_reconstruction_101NumDims, this.NumDims.x_reconstruction__97);

            % Mul:
            x_reconstruction_103 = x_reconstruction__87 .* x_reconstruction__96;
            x_reconstruction_103NumDims = max(x_reconstruction__87NumDims, x_reconstruction__96NumDims);

            % Set graph output arguments
            x_reconstruction_103NumDims1005 = x_reconstruction_103NumDims;

        end

    end

end