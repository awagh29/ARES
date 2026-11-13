classdef Squeeze_To_ReshapeLayer1005 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        reconstruction_mod_7
        reconstruction_mod_6
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
            this_cg = AresLidarONNX.coder.Squeeze_To_ReshapeLayer1005(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = AresLidarONNX.Squeeze_To_ReshapeLayer1005(cgInstance.Name);
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
            this_ml.reconstruction_mod_7 = cgInstance.reconstruction_mod_7;
            this_ml.reconstruction_mod_6 = cgInstance.reconstruction_mod_6;
        end
    end

    methods
        function this = Squeeze_To_ReshapeLayer1005(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_reconstruction__18'};
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
            this.reconstruction_mod_7 = mlInstance.reconstruction_mod_7;
            this.reconstruction_mod_6 = mlInstance.reconstruction_mod_6;
        end

        function [x_reconstruction__18] = predict(this, x_reconstruction_140__)
            if isdlarray(x_reconstruction_140__)
                x_reconstruction_140_ = stripdims(x_reconstruction_140__);
            else
                x_reconstruction_140_ = x_reconstruction_140__;
            end
            x_reconstruction_140NumDims = 4;
            x_reconstruction_140 = AresLidarONNX.coder.ops.permuteInputVar(x_reconstruction_140_, [3 4 2 1], 4);

            [x_reconstruction__18__, x_reconstruction__18NumDims__] = Squeeze_To_ReshapeGraph1012(this, x_reconstruction_140, x_reconstruction_140NumDims, false);
            x_reconstruction__18_ = AresLidarONNX.coder.ops.permuteOutputVar(x_reconstruction__18__, [3 4 2 1], 4);

            x_reconstruction__18 = dlarray(single(x_reconstruction__18_), 'SSCB');
        end

        function [x_reconstruction__18, x_reconstruction__18NumDims1016] = Squeeze_To_ReshapeGraph1012(this, x_reconstruction_140, x_reconstruction_140NumDims, Training)

            % Execute the operators:
            % Squeeze:
            [x_reconstruction_144, x_reconstruction_144NumDims] = AresLidarONNX.coder.ops.onnxSqueeze(x_reconstruction_140, this.Vars.x_reconstruction_137, coder.const(x_reconstruction_140NumDims));

            % Transpose:
            [perm1026, x_reconstruction_145NumDims] = AresLidarONNX.coder.ops.prepareTransposeArgs(this.Vars.TransposePerm1013, coder.const(x_reconstruction_144NumDims));
            if isempty(perm1026)
                x_reconstruction_145 = x_reconstruction_144;
            else
                x_reconstruction_145 = permute(AresLidarONNX.coder.ops.extractIfDlarray(x_reconstruction_144), perm1026);
            end

            % Gather:
            [x_reconstruction__16, x_reconstruction__16NumDims] = AresLidarONNX.coder.ops.onnxGather(x_reconstruction_145, this.Vars.x_reconstruction_122, 1, coder.const(x_reconstruction_145NumDims), this.NumDims.x_reconstruction_122);

            % Gemm:
            [A1027, B1028, C1029, alpha1030, beta1031, x_reconstruction__59NumDims] = AresLidarONNX.coder.ops.prepareGemmArgs(x_reconstruction__16, this.reconstruction_mod_7, this.reconstruction_mod_6, this.Vars.Gemmalpha1014, this.Vars.Gemmbeta1015, 0, 1, this.NumDims.reconstruction_mod_6);
            x_reconstruction__59 = alpha1030*B1028*A1027 + beta1031*C1029;

            % Div:
            x_reconstruction__64 = x_reconstruction__59 ./ this.Vars.x_reconstruction__63;
            x_reconstruction__64NumDims = max(coder.const(x_reconstruction__59NumDims), this.NumDims.x_reconstruction__63);

            % Erf:
            x_reconstruction__65 = erf(x_reconstruction__64);
            x_reconstruction__65NumDims = coder.const(x_reconstruction__64NumDims);

            % Add:
            x_reconstruction__60 = x_reconstruction__65 + this.Vars.x_reconstruction__61;
            x_reconstruction__60NumDims = max(coder.const(x_reconstruction__65NumDims), this.NumDims.x_reconstruction__61);

            % Mul:
            x_reconstruction__67 = x_reconstruction__59 .* x_reconstruction__60;
            x_reconstruction__67NumDims = max(coder.const(x_reconstruction__59NumDims), coder.const(x_reconstruction__60NumDims));

            % Mul:
            x_reconstruction__66 = x_reconstruction__67 .* this.Vars.x_reconstruction__62;
            x_reconstruction__66NumDims = max(coder.const(x_reconstruction__67NumDims), this.NumDims.x_reconstruction__62);

            % Reshape:
            [shape1032, x_reconstruction__18NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction__66, this.Vars.x_reconstruction_m_7, coder.const(x_reconstruction__66NumDims), 0);
            x_reconstruction__18 = reshape(x_reconstruction__66, shape1032{:});

            % Set graph output arguments
            x_reconstruction__18NumDims1016 = coder.const(x_reconstruction__18NumDims);

        end

    end

end