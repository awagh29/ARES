classdef Shape_To_MulLayer1007 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
            this_cg = AresLidarONNX.coder.Shape_To_MulLayer1007(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = AresLidarONNX.Shape_To_MulLayer1007(cgInstance.Name);
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
            this_ml.x_reconstruction__43 = cgInstance.x_reconstruction__43;
            this_ml.x_reconstruction__42 = cgInstance.x_reconstruction__42;
        end
    end

    methods
        function this = Shape_To_MulLayer1007(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_reconstruction__57'};
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
            this.x_reconstruction__43 = mlInstance.x_reconstruction__43;
            this.x_reconstruction__42 = mlInstance.x_reconstruction__42;
        end

        function [x_reconstruction__57] = predict(this, x_reconstruction__40__)
            if isdlarray(x_reconstruction__40__)
                x_reconstruction__40_ = stripdims(x_reconstruction__40__);
            else
                x_reconstruction__40_ = x_reconstruction__40__;
            end
            x_reconstruction__40NumDims = 4;
            x_reconstruction__40 = AresLidarONNX.coder.ops.permuteInputVar(x_reconstruction__40_, [4 3 1 2], 4);

            [x_reconstruction__57__, x_reconstruction__57NumDims__] = Shape_To_MulGraph1019(this, x_reconstruction__40, x_reconstruction__40NumDims, false);
            x_reconstruction__57_ = AresLidarONNX.coder.ops.permuteOutputVar(x_reconstruction__57__, [3 4 2 1], 4);

            x_reconstruction__57 = dlarray(single(x_reconstruction__57_), 'SSCB');
        end

        function [x_reconstruction__57, x_reconstruction__57NumDims1020] = Shape_To_MulGraph1019(this, x_reconstruction__40, x_reconstruction__40NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction__49, x_reconstruction__49NumDims] = AresLidarONNX.coder.ops.onnxShape(x_reconstruction__40, coder.const(x_reconstruction__40NumDims), 0, coder.const(x_reconstruction__40NumDims)+1);

            % Reshape:
            [shape1038, x_reconstruction__48NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction__40, this.Vars.x_reconstruction__44, coder.const(x_reconstruction__40NumDims), 0);
            x_reconstruction__48 = reshape(x_reconstruction__40, shape1038{:});

            % InstanceNormalization:
            if coder.const(x_reconstruction__48NumDims) > 2
                dataFormat1039 = coder.const([repmat('S',[1 coder.const(x_reconstruction__48NumDims)-2]),'CB']);
                X1040 = dlarray(AresLidarONNX.coder.ops.extractIfDlarray(x_reconstruction__48));
                Y1041 = nnet.internal.cnn.coder.layer.utils.normUtils.groupnorm(dlarray(X1040, dataFormat1039), numel(this.x_reconstruction__43), this.x_reconstruction__43, this.x_reconstruction__42, 0.000010);
                x_reconstruction__45 = AresLidarONNX.coder.ops.extractIfDlarray(Y1041);
            else
                x_reconstruction__45 = x_reconstruction__48;
            end
            x_reconstruction__45NumDims = coder.const(x_reconstruction__48NumDims);

            % Reshape:
            [shape1042, x_reconstruction__47NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction__45, x_reconstruction__49, coder.const(x_reconstruction__45NumDims), 0);
            x_reconstruction__47 = reshape(x_reconstruction__45, shape1042{:});

            % Mul:
            x_reconstruction__46 = x_reconstruction__47 .* this.Vars.onnx__Mul_354;
            x_reconstruction__46NumDims = max(coder.const(x_reconstruction__47NumDims), this.NumDims.onnx__Mul_354);

            % Add:
            x_reconstruction__41 = x_reconstruction__46 + this.Vars.onnx__Add_355;
            x_reconstruction__41NumDims = max(coder.const(x_reconstruction__46NumDims), this.NumDims.onnx__Add_355);

            % Div:
            x_reconstruction__54 = x_reconstruction__41 ./ this.Vars.x_reconstruction__53;
            x_reconstruction__54NumDims = max(coder.const(x_reconstruction__41NumDims), this.NumDims.x_reconstruction__53);

            % Erf:
            x_reconstruction__55 = erf(x_reconstruction__54);
            x_reconstruction__55NumDims = coder.const(x_reconstruction__54NumDims);

            % Add:
            x_reconstruction__50 = x_reconstruction__55 + this.Vars.x_reconstruction__51;
            x_reconstruction__50NumDims = max(coder.const(x_reconstruction__55NumDims), this.NumDims.x_reconstruction__51);

            % Mul:
            x_reconstruction__57 = x_reconstruction__41 .* x_reconstruction__50;
            x_reconstruction__57NumDims = max(coder.const(x_reconstruction__41NumDims), coder.const(x_reconstruction__50NumDims));

            % Set graph output arguments
            x_reconstruction__57NumDims1020 = coder.const(x_reconstruction__57NumDims);

        end

    end

end