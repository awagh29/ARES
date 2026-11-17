classdef Shape_To_MulLayer1002 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
            this_cg = AresLidarONNX.coder.Shape_To_MulLayer1002(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = AresLidarONNX.Shape_To_MulLayer1002(cgInstance.Name);
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
            this_ml.x_reconstruction__89 = cgInstance.x_reconstruction__89;
            this_ml.x_reconstruction__88 = cgInstance.x_reconstruction__88;
        end
    end

    methods
        function this = Shape_To_MulLayer1002(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_reconstruction_103'};
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
            this.x_reconstruction__89 = mlInstance.x_reconstruction__89;
            this.x_reconstruction__88 = mlInstance.x_reconstruction__88;
        end

        function [x_reconstruction_103] = predict(this, x_reconstruction__86__)
            if isdlarray(x_reconstruction__86__)
                x_reconstruction__86_ = stripdims(x_reconstruction__86__);
            else
                x_reconstruction__86_ = x_reconstruction__86__;
            end
            x_reconstruction__86NumDims = 4;
            x_reconstruction__86 = AresLidarONNX.coder.ops.permuteInputVar(x_reconstruction__86_, [4 3 1 2], 4);

            [x_reconstruction_103__, x_reconstruction_103NumDims__] = Shape_To_MulGraph1004(this, x_reconstruction__86, x_reconstruction__86NumDims, false);
            x_reconstruction_103_ = AresLidarONNX.coder.ops.permuteOutputVar(x_reconstruction_103__, [3 4 2 1], 4);

            x_reconstruction_103 = dlarray(single(x_reconstruction_103_), 'SSCB');
        end

        function [x_reconstruction_103, x_reconstruction_103NumDims1005] = Shape_To_MulGraph1004(this, x_reconstruction__86, x_reconstruction__86NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction__95, x_reconstruction__95NumDims] = AresLidarONNX.coder.ops.onnxShape(x_reconstruction__86, coder.const(x_reconstruction__86NumDims), 0, coder.const(x_reconstruction__86NumDims)+1);

            % Reshape:
            [shape1006, x_reconstruction__94NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction__86, this.Vars.x_reconstruction__90, coder.const(x_reconstruction__86NumDims), 0);
            x_reconstruction__94 = reshape(x_reconstruction__86, shape1006{:});

            % InstanceNormalization:
            if coder.const(x_reconstruction__94NumDims) > 2
                dataFormat1007 = coder.const([repmat('S',[1 coder.const(x_reconstruction__94NumDims)-2]),'CB']);
                X1008 = dlarray(AresLidarONNX.coder.ops.extractIfDlarray(x_reconstruction__94));
                Y1009 = nnet.internal.cnn.coder.layer.utils.normUtils.groupnorm(dlarray(X1008, dataFormat1007), numel(this.x_reconstruction__89), this.x_reconstruction__89, this.x_reconstruction__88, 0.000010);
                x_reconstruction__91 = AresLidarONNX.coder.ops.extractIfDlarray(Y1009);
            else
                x_reconstruction__91 = x_reconstruction__94;
            end
            x_reconstruction__91NumDims = coder.const(x_reconstruction__94NumDims);

            % Reshape:
            [shape1010, x_reconstruction__93NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction__91, x_reconstruction__95, coder.const(x_reconstruction__91NumDims), 0);
            x_reconstruction__93 = reshape(x_reconstruction__91, shape1010{:});

            % Mul:
            x_reconstruction__92 = x_reconstruction__93 .* this.Vars.onnx__Mul_318;
            x_reconstruction__92NumDims = max(coder.const(x_reconstruction__93NumDims), this.NumDims.onnx__Mul_318);

            % Add:
            x_reconstruction__87 = x_reconstruction__92 + this.Vars.onnx__Add_319;
            x_reconstruction__87NumDims = max(coder.const(x_reconstruction__92NumDims), this.NumDims.onnx__Add_319);

            % Div:
            x_reconstruction_100 = x_reconstruction__87 ./ this.Vars.x_reconstruction__99;
            x_reconstruction_100NumDims = max(coder.const(x_reconstruction__87NumDims), this.NumDims.x_reconstruction__99);

            % Erf:
            x_reconstruction_101 = erf(x_reconstruction_100);
            x_reconstruction_101NumDims = coder.const(x_reconstruction_100NumDims);

            % Add:
            x_reconstruction__96 = x_reconstruction_101 + this.Vars.x_reconstruction__97;
            x_reconstruction__96NumDims = max(coder.const(x_reconstruction_101NumDims), this.NumDims.x_reconstruction__97);

            % Mul:
            x_reconstruction_103 = x_reconstruction__87 .* x_reconstruction__96;
            x_reconstruction_103NumDims = max(coder.const(x_reconstruction__87NumDims), coder.const(x_reconstruction__96NumDims));

            % Set graph output arguments
            x_reconstruction_103NumDims1005 = coder.const(x_reconstruction_103NumDims);

        end

    end

end