classdef Shape_To_MulLayer1001 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = AresLidarONNX.coder.Shape_To_MulLayer1001(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = AresLidarONNX.Shape_To_MulLayer1001(cgInstance.Name);
            if isstruct(cgInstance.Vars)
                names = fieldnames(cgInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));
                end
            else
                this_ml.Vars = [];
            end
            this_ml.NumDims = cgInstance.NumDims;
            this_ml.x_reconstruction__71 = cgInstance.x_reconstruction__71;
            this_ml.x_reconstruction__70 = cgInstance.x_reconstruction__70;
        end
    end

    methods
        function this = Shape_To_MulLayer1001(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_reconstruction__85'};
            if isstruct(mlInstance.Vars)
                names = fieldnames(mlInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this.Vars.(fieldname) = AresLidarONNX.coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));
                end
            else
                this.Vars = [];
            end

            this.NumDims = mlInstance.NumDims;
            this.x_reconstruction__71 = mlInstance.x_reconstruction__71;
            this.x_reconstruction__70 = mlInstance.x_reconstruction__70;
        end

        function [x_reconstruction__85] = predict(this, x_reconstruction__68__)
            if isdlarray(x_reconstruction__68__)
                x_reconstruction__68_ = stripdims(x_reconstruction__68__);
            else
                x_reconstruction__68_ = x_reconstruction__68__;
            end
            x_reconstruction__68NumDims = 4;
            x_reconstruction__68 = AresLidarONNX.coder.ops.permuteInputVar(x_reconstruction__68_, [4 3 1 2], 4);

            [x_reconstruction__85__, x_reconstruction__85NumDims__] = Shape_To_MulGraph1002(this, x_reconstruction__68, x_reconstruction__68NumDims, false);
            x_reconstruction__85_ = AresLidarONNX.coder.ops.permuteOutputVar(x_reconstruction__85__, [3 4 2 1], 4);

            x_reconstruction__85 = dlarray(single(x_reconstruction__85_), 'SSCB');
        end

        function [x_reconstruction__85, x_reconstruction__85NumDims1003] = Shape_To_MulGraph1002(this, x_reconstruction__68, x_reconstruction__68NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction__77, x_reconstruction__77NumDims] = AresLidarONNX.coder.ops.onnxShape(x_reconstruction__68, coder.const(x_reconstruction__68NumDims), 0, coder.const(x_reconstruction__68NumDims)+1);

            % Reshape:
            [shape1001, x_reconstruction__76NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction__68, this.Vars.x_reconstruction__72, coder.const(x_reconstruction__68NumDims), 0);
            x_reconstruction__76 = reshape(x_reconstruction__68, shape1001{:});

            % InstanceNormalization:
            if coder.const(x_reconstruction__76NumDims) > 2
                dataFormat1002 = coder.const([repmat('S',[1 coder.const(x_reconstruction__76NumDims)-2]),'CB']);
                X1003 = dlarray(AresLidarONNX.coder.ops.extractIfDlarray(x_reconstruction__76));
                Y1004 = nnet.internal.cnn.coder.layer.utils.normUtils.groupnorm(dlarray(X1003, dataFormat1002), numel(this.x_reconstruction__71), this.x_reconstruction__71, this.x_reconstruction__70, 0.000010);
                x_reconstruction__73 = AresLidarONNX.coder.ops.extractIfDlarray(Y1004);
            else
                x_reconstruction__73 = x_reconstruction__76;
            end
            x_reconstruction__73NumDims = coder.const(x_reconstruction__76NumDims);

            % Reshape:
            [shape1005, x_reconstruction__75NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction__73, x_reconstruction__77, coder.const(x_reconstruction__73NumDims), 0);
            x_reconstruction__75 = reshape(x_reconstruction__73, shape1005{:});

            % Mul:
            x_reconstruction__74 = x_reconstruction__75 .* this.Vars.onnx__Mul_316;
            x_reconstruction__74NumDims = max(coder.const(x_reconstruction__75NumDims), this.NumDims.onnx__Mul_316);

            % Add:
            x_reconstruction__69 = x_reconstruction__74 + this.Vars.onnx__Add_317;
            x_reconstruction__69NumDims = max(coder.const(x_reconstruction__74NumDims), this.NumDims.onnx__Add_317);

            % Div:
            x_reconstruction__82 = x_reconstruction__69 ./ this.Vars.x_reconstruction__81;
            x_reconstruction__82NumDims = max(coder.const(x_reconstruction__69NumDims), this.NumDims.x_reconstruction__81);

            % Erf:
            x_reconstruction__83 = erf(x_reconstruction__82);
            x_reconstruction__83NumDims = coder.const(x_reconstruction__82NumDims);

            % Add:
            x_reconstruction__78 = x_reconstruction__83 + this.Vars.x_reconstruction__79;
            x_reconstruction__78NumDims = max(coder.const(x_reconstruction__83NumDims), this.NumDims.x_reconstruction__79);

            % Mul:
            x_reconstruction__85 = x_reconstruction__69 .* x_reconstruction__78;
            x_reconstruction__85NumDims = max(coder.const(x_reconstruction__69NumDims), coder.const(x_reconstruction__78NumDims));

            % Set graph output arguments
            x_reconstruction__85NumDims1003 = coder.const(x_reconstruction__85NumDims);

        end

    end

end