classdef Squeeze_To_ReshapeLayer1005 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

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
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'AresLidarONNX.coder.Squeeze_To_ReshapeLayer1005';
        end
    end


    methods
        function this = Squeeze_To_ReshapeLayer1005(name)
            this.Name = name;
            this.OutputNames = {'x_reconstruction__18'};
        end

        function [x_reconstruction__18] = predict(this, x_reconstruction_140)
            if isdlarray(x_reconstruction_140)
                x_reconstruction_140 = stripdims(x_reconstruction_140);
            end
            x_reconstruction_140NumDims = 4;
            x_reconstruction_140 = AresLidarONNX.ops.permuteInputVar(x_reconstruction_140, [3 4 2 1], 4);

            [x_reconstruction__18, x_reconstruction__18NumDims] = Squeeze_To_ReshapeGraph1012(this, x_reconstruction_140, x_reconstruction_140NumDims, false);
            x_reconstruction__18 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__18, [3 4 2 1], 4);

            x_reconstruction__18 = dlarray(single(x_reconstruction__18), 'SSCB');
        end

        function [x_reconstruction__18] = forward(this, x_reconstruction_140)
            if isdlarray(x_reconstruction_140)
                x_reconstruction_140 = stripdims(x_reconstruction_140);
            end
            x_reconstruction_140NumDims = 4;
            x_reconstruction_140 = AresLidarONNX.ops.permuteInputVar(x_reconstruction_140, [3 4 2 1], 4);

            [x_reconstruction__18, x_reconstruction__18NumDims] = Squeeze_To_ReshapeGraph1012(this, x_reconstruction_140, x_reconstruction_140NumDims, true);
            x_reconstruction__18 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__18, [3 4 2 1], 4);

            x_reconstruction__18 = dlarray(single(x_reconstruction__18), 'SSCB');
        end

        function [x_reconstruction__18, x_reconstruction__18NumDims1016] = Squeeze_To_ReshapeGraph1012(this, x_reconstruction_140, x_reconstruction_140NumDims, Training)

            % Execute the operators:
            % Squeeze:
            [x_reconstruction_144, x_reconstruction_144NumDims] = AresLidarONNX.ops.onnxSqueeze(x_reconstruction_140, this.Vars.x_reconstruction_137, x_reconstruction_140NumDims);

            % Transpose:
            [perm, x_reconstruction_145NumDims] = AresLidarONNX.ops.prepareTransposeArgs(this.Vars.TransposePerm1013, x_reconstruction_144NumDims);
            if isempty(perm)
                x_reconstruction_145 = x_reconstruction_144;
            else
                x_reconstruction_145 = permute(x_reconstruction_144, perm);
            end

            % Gather:
            [x_reconstruction__16, x_reconstruction__16NumDims] = AresLidarONNX.ops.onnxGather(x_reconstruction_145, this.Vars.x_reconstruction_122, 1, x_reconstruction_145NumDims, this.NumDims.x_reconstruction_122);

            % Gemm:
            [A, B, C, alpha, beta, x_reconstruction__59NumDims] = AresLidarONNX.ops.prepareGemmArgs(x_reconstruction__16, this.reconstruction_mod_7, this.reconstruction_mod_6, this.Vars.Gemmalpha1014, this.Vars.Gemmbeta1015, 0, 1, this.NumDims.reconstruction_mod_6);
            x_reconstruction__59 = alpha*B*A + beta*C;

            % Div:
            x_reconstruction__64 = x_reconstruction__59 ./ this.Vars.x_reconstruction__63;
            x_reconstruction__64NumDims = max(x_reconstruction__59NumDims, this.NumDims.x_reconstruction__63);

            % Erf:
            x_reconstruction__65 = erf(x_reconstruction__64);
            x_reconstruction__65NumDims = x_reconstruction__64NumDims;

            % Add:
            x_reconstruction__60 = x_reconstruction__65 + this.Vars.x_reconstruction__61;
            x_reconstruction__60NumDims = max(x_reconstruction__65NumDims, this.NumDims.x_reconstruction__61);

            % Mul:
            x_reconstruction__67 = x_reconstruction__59 .* x_reconstruction__60;
            x_reconstruction__67NumDims = max(x_reconstruction__59NumDims, x_reconstruction__60NumDims);

            % Mul:
            x_reconstruction__66 = x_reconstruction__67 .* this.Vars.x_reconstruction__62;
            x_reconstruction__66NumDims = max(x_reconstruction__67NumDims, this.NumDims.x_reconstruction__62);

            % Reshape:
            [shape, x_reconstruction__18NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction__66, this.Vars.x_reconstruction_m_7, x_reconstruction__66NumDims, 0);
            x_reconstruction__18 = reshape(x_reconstruction__66, shape{:});

            % Set graph output arguments
            x_reconstruction__18NumDims1016 = x_reconstruction__18NumDims;

        end

    end

end