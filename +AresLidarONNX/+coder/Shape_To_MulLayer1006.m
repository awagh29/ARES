classdef Shape_To_MulLayer1006 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
            this_cg = AresLidarONNX.coder.Shape_To_MulLayer1006(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = AresLidarONNX.Shape_To_MulLayer1006(cgInstance.Name);
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
            this_ml.x_reconstruction__25 = cgInstance.x_reconstruction__25;
            this_ml.x_reconstruction__24 = cgInstance.x_reconstruction__24;
        end
    end

    methods
        function this = Shape_To_MulLayer1006(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_reconstruction__39'};
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
            this.x_reconstruction__25 = mlInstance.x_reconstruction__25;
            this.x_reconstruction__24 = mlInstance.x_reconstruction__24;
        end

        function [x_reconstruction__39] = predict(this, x_reconstruction__22__)
            if isdlarray(x_reconstruction__22__)
                x_reconstruction__22_ = stripdims(x_reconstruction__22__);
            else
                x_reconstruction__22_ = x_reconstruction__22__;
            end
            x_reconstruction__22NumDims = 4;
            x_reconstruction__22 = AresLidarONNX.coder.ops.permuteInputVar(x_reconstruction__22_, [4 3 1 2], 4);

            [x_reconstruction__39__, x_reconstruction__39NumDims__] = Shape_To_MulGraph1017(this, x_reconstruction__22, x_reconstruction__22NumDims, false);
            x_reconstruction__39_ = AresLidarONNX.coder.ops.permuteOutputVar(x_reconstruction__39__, [3 4 2 1], 4);

            x_reconstruction__39 = dlarray(single(x_reconstruction__39_), 'SSCB');
        end

        function [x_reconstruction__39, x_reconstruction__39NumDims1018] = Shape_To_MulGraph1017(this, x_reconstruction__22, x_reconstruction__22NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction__31, x_reconstruction__31NumDims] = AresLidarONNX.coder.ops.onnxShape(x_reconstruction__22, coder.const(x_reconstruction__22NumDims), 0, coder.const(x_reconstruction__22NumDims)+1);

            % Reshape:
            [shape1033, x_reconstruction__30NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction__22, this.Vars.x_reconstruction__26, coder.const(x_reconstruction__22NumDims), 0);
            x_reconstruction__30 = reshape(x_reconstruction__22, shape1033{:});

            % InstanceNormalization:
            if coder.const(x_reconstruction__30NumDims) > 2
                dataFormat1034 = coder.const([repmat('S',[1 coder.const(x_reconstruction__30NumDims)-2]),'CB']);
                X1035 = dlarray(AresLidarONNX.coder.ops.extractIfDlarray(x_reconstruction__30));
                Y1036 = nnet.internal.cnn.coder.layer.utils.normUtils.groupnorm(dlarray(X1035, dataFormat1034), numel(this.x_reconstruction__25), this.x_reconstruction__25, this.x_reconstruction__24, 0.000010);
                x_reconstruction__27 = AresLidarONNX.coder.ops.extractIfDlarray(Y1036);
            else
                x_reconstruction__27 = x_reconstruction__30;
            end
            x_reconstruction__27NumDims = coder.const(x_reconstruction__30NumDims);

            % Reshape:
            [shape1037, x_reconstruction__29NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction__27, x_reconstruction__31, coder.const(x_reconstruction__27NumDims), 0);
            x_reconstruction__29 = reshape(x_reconstruction__27, shape1037{:});

            % Mul:
            x_reconstruction__28 = x_reconstruction__29 .* this.Vars.onnx__Mul_352;
            x_reconstruction__28NumDims = max(coder.const(x_reconstruction__29NumDims), this.NumDims.onnx__Mul_352);

            % Add:
            x_reconstruction__23 = x_reconstruction__28 + this.Vars.onnx__Add_353;
            x_reconstruction__23NumDims = max(coder.const(x_reconstruction__28NumDims), this.NumDims.onnx__Add_353);

            % Div:
            x_reconstruction__36 = x_reconstruction__23 ./ this.Vars.x_reconstruction__35;
            x_reconstruction__36NumDims = max(coder.const(x_reconstruction__23NumDims), this.NumDims.x_reconstruction__35);

            % Erf:
            x_reconstruction__37 = erf(x_reconstruction__36);
            x_reconstruction__37NumDims = coder.const(x_reconstruction__36NumDims);

            % Add:
            x_reconstruction__32 = x_reconstruction__37 + this.Vars.x_reconstruction__33;
            x_reconstruction__32NumDims = max(coder.const(x_reconstruction__37NumDims), this.NumDims.x_reconstruction__33);

            % Mul:
            x_reconstruction__39 = x_reconstruction__23 .* x_reconstruction__32;
            x_reconstruction__39NumDims = max(coder.const(x_reconstruction__23NumDims), coder.const(x_reconstruction__32NumDims));

            % Set graph output arguments
            x_reconstruction__39NumDims1018 = coder.const(x_reconstruction__39NumDims);

        end

    end

end