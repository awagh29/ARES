classdef Shape_To_ReshapeLayer1003 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        x_reconstruction_107
        x_reconstruction_106
        reconstruction_mo_15
        reconstruction_mo_14
        reconstruction_mo_17
        reconstruction_mo_16
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
            name = 'AresLidarONNX.coder.Shape_To_ReshapeLayer1003';
        end
    end


    methods
        function this = Shape_To_ReshapeLayer1003(name)
            this.Name = name;
            this.OutputNames = {'x_reconstruction__17'};
        end

        function [x_reconstruction__17] = predict(this, x_reconstruction_104)
            if isdlarray(x_reconstruction_104)
                x_reconstruction_104 = stripdims(x_reconstruction_104);
            end
            x_reconstruction_104NumDims = 4;
            x_reconstruction_104 = AresLidarONNX.ops.permuteInputVar(x_reconstruction_104, [4 3 1 2], 4);

            [x_reconstruction__17, x_reconstruction__17NumDims] = Shape_To_ReshapeGraph1006(this, x_reconstruction_104, x_reconstruction_104NumDims, false);
            x_reconstruction__17 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__17, [3 1 2], 3);

            x_reconstruction__17 = dlarray(single(x_reconstruction__17), 'CBT');
        end

        function [x_reconstruction__17] = forward(this, x_reconstruction_104)
            if isdlarray(x_reconstruction_104)
                x_reconstruction_104 = stripdims(x_reconstruction_104);
            end
            x_reconstruction_104NumDims = 4;
            x_reconstruction_104 = AresLidarONNX.ops.permuteInputVar(x_reconstruction_104, [4 3 1 2], 4);

            [x_reconstruction__17, x_reconstruction__17NumDims] = Shape_To_ReshapeGraph1006(this, x_reconstruction_104, x_reconstruction_104NumDims, true);
            x_reconstruction__17 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__17, [3 1 2], 3);

            x_reconstruction__17 = dlarray(single(x_reconstruction__17), 'CBT');
        end

        function [x_reconstruction__17, x_reconstruction__17NumDims1009] = Shape_To_ReshapeGraph1006(this, x_reconstruction_104, x_reconstruction_104NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction_113, x_reconstruction_113NumDims] = AresLidarONNX.ops.onnxShape(x_reconstruction_104, x_reconstruction_104NumDims, 0, x_reconstruction_104NumDims+1);

            % Reshape:
            [shape, x_reconstruction_112NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction_104, this.Vars.x_reconstruction_108, x_reconstruction_104NumDims, 0);
            x_reconstruction_112 = reshape(x_reconstruction_104, shape{:});

            % InstanceNormalization:
            if x_reconstruction_112NumDims > 2
                dataFormat = [repmat('S',[1 x_reconstruction_112NumDims-2]),'CB'];
                x_reconstruction_109 = instancenorm(x_reconstruction_112, this.x_reconstruction_107, this.x_reconstruction_106, 'Epsilon', 0.000010, 'DataFormat', dataFormat);
            else
                x_reconstruction_109 = x_reconstruction_112;
            end
            x_reconstruction_109NumDims = x_reconstruction_112NumDims;

            % Reshape:
            [shape, x_reconstruction_111NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction_109, x_reconstruction_113, x_reconstruction_109NumDims, 0);
            x_reconstruction_111 = reshape(x_reconstruction_109, shape{:});

            % Mul:
            x_reconstruction_110 = x_reconstruction_111 .* this.Vars.onnx__Mul_320;
            x_reconstruction_110NumDims = max(x_reconstruction_111NumDims, this.NumDims.onnx__Mul_320);

            % Add:
            x_reconstruction_105 = x_reconstruction_110 + this.Vars.onnx__Add_321;
            x_reconstruction_105NumDims = max(x_reconstruction_110NumDims, this.NumDims.onnx__Add_321);

            % Div:
            x_reconstruction_118 = x_reconstruction_105 ./ this.Vars.x_reconstruction_117;
            x_reconstruction_118NumDims = max(x_reconstruction_105NumDims, this.NumDims.x_reconstruction_117);

            % Erf:
            x_reconstruction_119 = erf(x_reconstruction_118);
            x_reconstruction_119NumDims = x_reconstruction_118NumDims;

            % Add:
            x_reconstruction_114 = x_reconstruction_119 + this.Vars.x_reconstruction_115;
            x_reconstruction_114NumDims = max(x_reconstruction_119NumDims, this.NumDims.x_reconstruction_115);

            % Mul:
            x_reconstruction_121 = x_reconstruction_105 .* x_reconstruction_114;
            x_reconstruction_121NumDims = max(x_reconstruction_105NumDims, x_reconstruction_114NumDims);

            % Mul:
            x_reconstruction_120 = x_reconstruction_121 .* this.Vars.x_reconstruction_116;
            x_reconstruction_120NumDims = max(x_reconstruction_121NumDims, this.NumDims.x_reconstruction_116);

            % Flatten:
            [dim1, dim2, x_reconstruction_123NumDims] = AresLidarONNX.ops.prepareFlattenArgs(x_reconstruction_120, 1, x_reconstruction_120NumDims);
            x_reconstruction_123 = reshape(x_reconstruction_120, dim1, dim2);

            % Gemm:
            [A, B, C, alpha, beta, x_reconstruction_124NumDims] = AresLidarONNX.ops.prepareGemmArgs(x_reconstruction_123, this.reconstruction_mo_15, this.reconstruction_mo_14, this.Vars.Gemmalpha1007, this.Vars.Gemmbeta1008, 0, 1, this.NumDims.reconstruction_mo_14);
            x_reconstruction_124 = alpha*B*A + beta*C;

            % Div:
            x_reconstruction_129 = x_reconstruction_124 ./ this.Vars.x_reconstruction_128;
            x_reconstruction_129NumDims = max(x_reconstruction_124NumDims, this.NumDims.x_reconstruction_128);

            % Erf:
            x_reconstruction_130 = erf(x_reconstruction_129);
            x_reconstruction_130NumDims = x_reconstruction_129NumDims;

            % Add:
            x_reconstruction_125 = x_reconstruction_130 + this.Vars.x_reconstruction_126;
            x_reconstruction_125NumDims = max(x_reconstruction_130NumDims, this.NumDims.x_reconstruction_126);

            % Mul:
            x_reconstruction_132 = x_reconstruction_124 .* x_reconstruction_125;
            x_reconstruction_132NumDims = max(x_reconstruction_124NumDims, x_reconstruction_125NumDims);

            % Mul:
            x_reconstruction_131 = x_reconstruction_132 .* this.Vars.x_reconstruction_127;
            x_reconstruction_131NumDims = max(x_reconstruction_132NumDims, this.NumDims.x_reconstruction_127);

            % LayerNormalization:
            [x_reconstruction_133, x_reconstruction_133NumDims] = AresLidarONNX.ops.onnxLayerNormalization(x_reconstruction_131, this.reconstruction_mo_17, this.reconstruction_mo_16, -1, 1.000000e-05, x_reconstruction_131NumDims);

            % Reshape:
            [shape, x_reconstruction__17NumDims] = AresLidarONNX.ops.prepareReshapeArgs(x_reconstruction_133, this.Vars.x_reconstruction_m_6, x_reconstruction_133NumDims, 0);
            x_reconstruction__17 = reshape(x_reconstruction_133, shape{:});

            % Set graph output arguments
            x_reconstruction__17NumDims1009 = x_reconstruction__17NumDims;

        end

    end

end