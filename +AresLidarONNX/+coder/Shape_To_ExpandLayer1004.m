classdef Shape_To_ExpandLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
            this_cg = AresLidarONNX.coder.Shape_To_ExpandLayer1004(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = AresLidarONNX.Shape_To_ExpandLayer1004(cgInstance.Name);
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
        end
    end

    methods
        function this = Shape_To_ExpandLayer1004(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_reconstruction_139'};
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
        end

        function [x_reconstruction_139] = predict(this, x_reconstruction_146__)
            if isdlarray(x_reconstruction_146__)
                x_reconstruction_146_ = stripdims(x_reconstruction_146__);
            else
                x_reconstruction_146_ = x_reconstruction_146__;
            end
            x_reconstruction_146NumDims = 3;
            x_reconstruction_146 = AresLidarONNX.coder.ops.permuteInputVar(x_reconstruction_146_, [3 2 1], 3);

            [x_reconstruction_139__, x_reconstruction_139NumDims__] = Shape_To_ExpandGraph1010(this, x_reconstruction_146, x_reconstruction_146NumDims, false);
            x_reconstruction_139_ = AresLidarONNX.coder.ops.permuteOutputVar(x_reconstruction_139__, [3 2 1], 3);

            x_reconstruction_139 = dlarray(single(x_reconstruction_139_), 'CB');
        end

        function [x_reconstruction_139, x_reconstruction_139NumDims1011] = Shape_To_ExpandGraph1010(this, x_reconstruction_146, x_reconstruction_146NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction_143, x_reconstruction_143NumDims] = AresLidarONNX.coder.ops.onnxShape(x_reconstruction_146, coder.const(x_reconstruction_146NumDims), 0, coder.const(x_reconstruction_146NumDims)+1);

            % Gather:
            [x_reconstruction_142, x_reconstruction_142NumDims] = AresLidarONNX.coder.ops.onnxGather(x_reconstruction_143, this.Vars.x_reconstruction_135, 0, coder.const(x_reconstruction_143NumDims), this.NumDims.x_reconstruction_135);

            % Unsqueeze:
            [shape1024, onnx__Concat_208NumDims] = AresLidarONNX.coder.ops.prepareUnsqueezeArgs(x_reconstruction_142, this.Vars.onnx__Unsqueeze_207, coder.const(x_reconstruction_142NumDims));
            onnx__Concat_208 = reshape(x_reconstruction_142, shape1024);

            % Concat:
            [x_reconstruction_134, x_reconstruction_134NumDims] = AresLidarONNX.coder.ops.onnxConcat(0, {this.Vars.onnx__Concat_346, onnx__Concat_208, this.Vars.x_reconstruction_136}, [this.NumDims.onnx__Concat_346, coder.const(onnx__Concat_208NumDims), this.NumDims.x_reconstruction_136]);

            % Expand:
            [shape1025, x_reconstruction_139NumDims] = AresLidarONNX.coder.ops.prepareExpandArgs(this.NumDims.x_reconstruction_138, x_reconstruction_134);
            x_reconstruction_139 = this.Vars.x_reconstruction_138 + zeros(shape1025);

            % Set graph output arguments
            x_reconstruction_139NumDims1011 = coder.const(x_reconstruction_139NumDims);

        end

    end

end