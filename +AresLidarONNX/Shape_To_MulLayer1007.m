classdef Shape_To_MulLayer1007 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        x_reconstruction__43
        x_reconstruction__42
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
            name = 'AresLidarONNX.coder.Shape_To_MulLayer1007';
        end
    end


    methods
        function this = Shape_To_MulLayer1007(name)
            this.Name = name;
            this.OutputNames = {'x_reconstruction__57'};
        end

        function [x_reconstruction__57] = predict(this, x_reconstruction__40)
            if isdlarray(x_reconstruction__40)
                x_reconstruction__40 = stripdims(x_reconstruction__40);
            end
            x_reconstruction__40NumDims = 4;
            x_reconstruction__40 = AresLidarONNX.ops.permuteInputVar(x_reconstruction__40, [4 3 1 2], 4);

            [x_reconstruction__57, x_reconstruction__57NumDims] = Shape_To_MulGraph1019(this, x_reconstruction__40, x_reconstruction__40NumDims, false);
            x_reconstruction__57 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__57, [3 4 2 1], 4);

            x_reconstruction__57 = dlarray(single(x_reconstruction__57), 'SSCB');
        end

        function [x_reconstruction__57] = forward(this, x_reconstruction__40)
            if isdlarray(x_reconstruction__40)
                x_reconstruction__40 = stripdims(x_reconstruction__40);
            end
            x_reconstruction__40NumDims = 4;
            x_reconstruction__40 = AresLidarONNX.ops.permuteInputVar(x_reconstruction__40, [4 3 1 2], 4);

            [x_reconstruction__57, x_reconstruction__57NumDims] = Shape_To_MulGraph1019(this, x_reconstruction__40, x_reconstruction__40NumDims, true);
            x_reconstruction__57 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__57, [3 4 2 1], 4);

            x_reconstruction__57 = dlarray(single(x_reconstruction__57), 'SSCB');
        end

        function [x_reconstruction__57, x_reconstruction__57NumDims1020] = Shape_To_MulGraph1019(this, x_reconstruction__40, x_reconstruction__40NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction__49, x_reconstruction__49NumDims] = AresLidarONNX.ops.onnxShape(x_reconstruction__40, x_reconstruction__40NumDims, 0, x_reconstruction__40NumDims+1);

            % Reshape:
            [shape, x_reconstruction__48NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction__40, this.Vars.x_reconstruction__44, x_reconstruction__40NumDims, 0);
            x_reconstruction__48 = reshape(x_reconstruction__40, shape{:});

            % InstanceNormalization:
            if x_reconstruction__48NumDims > 2
                dataFormat = [repmat('S',[1 x_reconstruction__48NumDims-2]),'CB'];
                x_reconstruction__45 = instancenorm(x_reconstruction__48, this.x_reconstruction__43, this.x_reconstruction__42, 'Epsilon', 0.000010, 'DataFormat', dataFormat);
            else
                x_reconstruction__45 = x_reconstruction__48;
            end
            x_reconstruction__45NumDims = x_reconstruction__48NumDims;

            % Reshape:
            [shape, x_reconstruction__47NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction__45, x_reconstruction__49, x_reconstruction__45NumDims, 0);
            x_reconstruction__47 = reshape(x_reconstruction__45, shape{:});

            % Mul:
            x_reconstruction__46 = x_reconstruction__47 .* this.Vars.onnx__Mul_354;
            x_reconstruction__46NumDims = max(x_reconstruction__47NumDims, this.NumDims.onnx__Mul_354);

            % Add:
            x_reconstruction__41 = x_reconstruction__46 + this.Vars.onnx__Add_355;
            x_reconstruction__41NumDims = max(x_reconstruction__46NumDims, this.NumDims.onnx__Add_355);

            % Div:
            x_reconstruction__54 = x_reconstruction__41 ./ this.Vars.x_reconstruction__53;
            x_reconstruction__54NumDims = max(x_reconstruction__41NumDims, this.NumDims.x_reconstruction__53);

            % Erf:
            x_reconstruction__55 = erf(x_reconstruction__54);
            x_reconstruction__55NumDims = x_reconstruction__54NumDims;

            % Add:
            x_reconstruction__50 = x_reconstruction__55 + this.Vars.x_reconstruction__51;
            x_reconstruction__50NumDims = max(x_reconstruction__55NumDims, this.NumDims.x_reconstruction__51);

            % Mul:
            x_reconstruction__57 = x_reconstruction__41 .* x_reconstruction__50;
            x_reconstruction__57NumDims = max(x_reconstruction__41NumDims, x_reconstruction__50NumDims);

            % Set graph output arguments
            x_reconstruction__57NumDims1020 = x_reconstruction__57NumDims;

        end

    end

end