classdef Shape_To_MulLayer1006 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        x_reconstruction__25
        x_reconstruction__24
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
            name = 'AresLidarONNX.coder.Shape_To_MulLayer1006';
        end
    end


    methods
        function this = Shape_To_MulLayer1006(name)
            this.Name = name;
            this.OutputNames = {'x_reconstruction__39'};
        end

        function [x_reconstruction__39] = predict(this, x_reconstruction__22)
            if isdlarray(x_reconstruction__22)
                x_reconstruction__22 = stripdims(x_reconstruction__22);
            end
            x_reconstruction__22NumDims = 4;
            x_reconstruction__22 = AresLidarONNX.ops.permuteInputVar(x_reconstruction__22, [4 3 1 2], 4);

            [x_reconstruction__39, x_reconstruction__39NumDims] = Shape_To_MulGraph1017(this, x_reconstruction__22, x_reconstruction__22NumDims, false);
            x_reconstruction__39 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__39, [3 4 2 1], 4);

            x_reconstruction__39 = dlarray(single(x_reconstruction__39), 'SSCB');
        end

        function [x_reconstruction__39] = forward(this, x_reconstruction__22)
            if isdlarray(x_reconstruction__22)
                x_reconstruction__22 = stripdims(x_reconstruction__22);
            end
            x_reconstruction__22NumDims = 4;
            x_reconstruction__22 = AresLidarONNX.ops.permuteInputVar(x_reconstruction__22, [4 3 1 2], 4);

            [x_reconstruction__39, x_reconstruction__39NumDims] = Shape_To_MulGraph1017(this, x_reconstruction__22, x_reconstruction__22NumDims, true);
            x_reconstruction__39 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__39, [3 4 2 1], 4);

            x_reconstruction__39 = dlarray(single(x_reconstruction__39), 'SSCB');
        end

        function [x_reconstruction__39, x_reconstruction__39NumDims1018] = Shape_To_MulGraph1017(this, x_reconstruction__22, x_reconstruction__22NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction__31, x_reconstruction__31NumDims] = AresLidarONNX.ops.onnxShape(x_reconstruction__22, x_reconstruction__22NumDims, 0, x_reconstruction__22NumDims+1);

            % Reshape:
            [shape, x_reconstruction__30NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction__22, this.Vars.x_reconstruction__26, x_reconstruction__22NumDims, 0);
            x_reconstruction__30 = reshape(x_reconstruction__22, shape{:});

            % InstanceNormalization:
            if x_reconstruction__30NumDims > 2
                dataFormat = [repmat('S',[1 x_reconstruction__30NumDims-2]),'CB'];
                x_reconstruction__27 = instancenorm(x_reconstruction__30, this.x_reconstruction__25, this.x_reconstruction__24, 'Epsilon', 0.000010, 'DataFormat', dataFormat);
            else
                x_reconstruction__27 = x_reconstruction__30;
            end
            x_reconstruction__27NumDims = x_reconstruction__30NumDims;

            % Reshape:
            [shape, x_reconstruction__29NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction__27, x_reconstruction__31, x_reconstruction__27NumDims, 0);
            x_reconstruction__29 = reshape(x_reconstruction__27, shape{:});

            % Mul:
            x_reconstruction__28 = x_reconstruction__29 .* this.Vars.onnx__Mul_352;
            x_reconstruction__28NumDims = max(x_reconstruction__29NumDims, this.NumDims.onnx__Mul_352);

            % Add:
            x_reconstruction__23 = x_reconstruction__28 + this.Vars.onnx__Add_353;
            x_reconstruction__23NumDims = max(x_reconstruction__28NumDims, this.NumDims.onnx__Add_353);

            % Div:
            x_reconstruction__36 = x_reconstruction__23 ./ this.Vars.x_reconstruction__35;
            x_reconstruction__36NumDims = max(x_reconstruction__23NumDims, this.NumDims.x_reconstruction__35);

            % Erf:
            x_reconstruction__37 = erf(x_reconstruction__36);
            x_reconstruction__37NumDims = x_reconstruction__36NumDims;

            % Add:
            x_reconstruction__32 = x_reconstruction__37 + this.Vars.x_reconstruction__33;
            x_reconstruction__32NumDims = max(x_reconstruction__37NumDims, this.NumDims.x_reconstruction__33);

            % Mul:
            x_reconstruction__39 = x_reconstruction__23 .* x_reconstruction__32;
            x_reconstruction__39NumDims = max(x_reconstruction__23NumDims, x_reconstruction__32NumDims);

            % Set graph output arguments
            x_reconstruction__39NumDims1018 = x_reconstruction__39NumDims;

        end

    end

end