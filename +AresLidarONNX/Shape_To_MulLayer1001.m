classdef Shape_To_MulLayer1001 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        x_reconstruction__71
        x_reconstruction__70
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
            name = 'AresLidarONNX.coder.Shape_To_MulLayer1001';
        end
    end


    methods
        function this = Shape_To_MulLayer1001(name)
            this.Name = name;
            this.OutputNames = {'x_reconstruction__85'};
        end

        function [x_reconstruction__85] = predict(this, x_reconstruction__68)
            if isdlarray(x_reconstruction__68)
                x_reconstruction__68 = stripdims(x_reconstruction__68);
            end
            x_reconstruction__68NumDims = 4;
            x_reconstruction__68 = AresLidarONNX.ops.permuteInputVar(x_reconstruction__68, [4 3 1 2], 4);

            [x_reconstruction__85, x_reconstruction__85NumDims] = Shape_To_MulGraph1002(this, x_reconstruction__68, x_reconstruction__68NumDims, false);
            x_reconstruction__85 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__85, [3 4 2 1], 4);

            x_reconstruction__85 = dlarray(single(x_reconstruction__85), 'SSCB');
        end

        function [x_reconstruction__85] = forward(this, x_reconstruction__68)
            if isdlarray(x_reconstruction__68)
                x_reconstruction__68 = stripdims(x_reconstruction__68);
            end
            x_reconstruction__68NumDims = 4;
            x_reconstruction__68 = AresLidarONNX.ops.permuteInputVar(x_reconstruction__68, [4 3 1 2], 4);

            [x_reconstruction__85, x_reconstruction__85NumDims] = Shape_To_MulGraph1002(this, x_reconstruction__68, x_reconstruction__68NumDims, true);
            x_reconstruction__85 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__85, [3 4 2 1], 4);

            x_reconstruction__85 = dlarray(single(x_reconstruction__85), 'SSCB');
        end

        function [x_reconstruction__85, x_reconstruction__85NumDims1003] = Shape_To_MulGraph1002(this, x_reconstruction__68, x_reconstruction__68NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction__77, x_reconstruction__77NumDims] = AresLidarONNX.ops.onnxShape(x_reconstruction__68, x_reconstruction__68NumDims, 0, x_reconstruction__68NumDims+1);

            % Reshape:
            [shape, x_reconstruction__76NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction__68, this.Vars.x_reconstruction__72, x_reconstruction__68NumDims, 0);
            x_reconstruction__76 = reshape(x_reconstruction__68, shape{:});

            % InstanceNormalization:
            if x_reconstruction__76NumDims > 2
                dataFormat = [repmat('S',[1 x_reconstruction__76NumDims-2]),'CB'];
                x_reconstruction__73 = instancenorm(x_reconstruction__76, this.x_reconstruction__71, this.x_reconstruction__70, 'Epsilon', 0.000010, 'DataFormat', dataFormat);
            else
                x_reconstruction__73 = x_reconstruction__76;
            end
            x_reconstruction__73NumDims = x_reconstruction__76NumDims;

            % Reshape:
            [shape, x_reconstruction__75NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction__73, x_reconstruction__77, x_reconstruction__73NumDims, 0);
            x_reconstruction__75 = reshape(x_reconstruction__73, shape{:});

            % Mul:
            x_reconstruction__74 = x_reconstruction__75 .* this.Vars.onnx__Mul_316;
            x_reconstruction__74NumDims = max(x_reconstruction__75NumDims, this.NumDims.onnx__Mul_316);

            % Add:
            x_reconstruction__69 = x_reconstruction__74 + this.Vars.onnx__Add_317;
            x_reconstruction__69NumDims = max(x_reconstruction__74NumDims, this.NumDims.onnx__Add_317);

            % Div:
            x_reconstruction__82 = x_reconstruction__69 ./ this.Vars.x_reconstruction__81;
            x_reconstruction__82NumDims = max(x_reconstruction__69NumDims, this.NumDims.x_reconstruction__81);

            % Erf:
            x_reconstruction__83 = erf(x_reconstruction__82);
            x_reconstruction__83NumDims = x_reconstruction__82NumDims;

            % Add:
            x_reconstruction__78 = x_reconstruction__83 + this.Vars.x_reconstruction__79;
            x_reconstruction__78NumDims = max(x_reconstruction__83NumDims, this.NumDims.x_reconstruction__79);

            % Mul:
            x_reconstruction__85 = x_reconstruction__69 .* x_reconstruction__78;
            x_reconstruction__85NumDims = max(x_reconstruction__69NumDims, x_reconstruction__78NumDims);

            % Set graph output arguments
            x_reconstruction__85NumDims1003 = x_reconstruction__85NumDims;

        end

    end

end