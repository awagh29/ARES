classdef Shape_To_ReshapeLayer1003 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
            this_cg = AresLidarONNX.coder.Shape_To_ReshapeLayer1003(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = AresLidarONNX.Shape_To_ReshapeLayer1003(cgInstance.Name);
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
            this_ml.x_reconstruction_107 = cgInstance.x_reconstruction_107;
            this_ml.x_reconstruction_106 = cgInstance.x_reconstruction_106;
            this_ml.reconstruction_mo_15 = cgInstance.reconstruction_mo_15;
            this_ml.reconstruction_mo_14 = cgInstance.reconstruction_mo_14;
            this_ml.reconstruction_mo_17 = cgInstance.reconstruction_mo_17;
            this_ml.reconstruction_mo_16 = cgInstance.reconstruction_mo_16;
        end
    end

    methods
        function this = Shape_To_ReshapeLayer1003(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_reconstruction__17'};
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
            this.x_reconstruction_107 = mlInstance.x_reconstruction_107;
            this.x_reconstruction_106 = mlInstance.x_reconstruction_106;
            this.reconstruction_mo_15 = mlInstance.reconstruction_mo_15;
            this.reconstruction_mo_14 = mlInstance.reconstruction_mo_14;
            this.reconstruction_mo_17 = mlInstance.reconstruction_mo_17;
            this.reconstruction_mo_16 = mlInstance.reconstruction_mo_16;
        end

        function [x_reconstruction__17] = predict(this, x_reconstruction_104__)
            if isdlarray(x_reconstruction_104__)
                x_reconstruction_104_ = stripdims(x_reconstruction_104__);
            else
                x_reconstruction_104_ = x_reconstruction_104__;
            end
            x_reconstruction_104NumDims = 4;
            x_reconstruction_104 = AresLidarONNX.coder.ops.permuteInputVar(x_reconstruction_104_, [4 3 1 2], 4);

            [x_reconstruction__17__, x_reconstruction__17NumDims__] = Shape_To_ReshapeGraph1006(this, x_reconstruction_104, x_reconstruction_104NumDims, false);
            x_reconstruction__17_ = AresLidarONNX.coder.ops.permuteOutputVar(x_reconstruction__17__, [3 1 2], 3);

            x_reconstruction__17 = dlarray(single(x_reconstruction__17_), 'CBT');
        end

        function [x_reconstruction__17, x_reconstruction__17NumDims1009] = Shape_To_ReshapeGraph1006(this, x_reconstruction_104, x_reconstruction_104NumDims, Training)

            % Execute the operators:
            % Shape:
            [x_reconstruction_113, x_reconstruction_113NumDims] = AresLidarONNX.coder.ops.onnxShape(x_reconstruction_104, coder.const(x_reconstruction_104NumDims), 0, coder.const(x_reconstruction_104NumDims)+1);

            % Reshape:
            [shape1011, x_reconstruction_112NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction_104, this.Vars.x_reconstruction_108, coder.const(x_reconstruction_104NumDims), 0);
            x_reconstruction_112 = reshape(x_reconstruction_104, shape1011{:});

            % InstanceNormalization:
            if coder.const(x_reconstruction_112NumDims) > 2
                dataFormat1012 = coder.const([repmat('S',[1 coder.const(x_reconstruction_112NumDims)-2]),'CB']);
                X1013 = dlarray(AresLidarONNX.coder.ops.extractIfDlarray(x_reconstruction_112));
                Y1014 = nnet.internal.cnn.coder.layer.utils.normUtils.groupnorm(dlarray(X1013, dataFormat1012), numel(this.x_reconstruction_107), this.x_reconstruction_107, this.x_reconstruction_106, 0.000010);
                x_reconstruction_109 = AresLidarONNX.coder.ops.extractIfDlarray(Y1014);
            else
                x_reconstruction_109 = x_reconstruction_112;
            end
            x_reconstruction_109NumDims = coder.const(x_reconstruction_112NumDims);

            % Reshape:
            [shape1015, x_reconstruction_111NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction_109, x_reconstruction_113, coder.const(x_reconstruction_109NumDims), 0);
            x_reconstruction_111 = reshape(x_reconstruction_109, shape1015{:});

            % Mul:
            x_reconstruction_110 = x_reconstruction_111 .* this.Vars.onnx__Mul_320;
            x_reconstruction_110NumDims = max(coder.const(x_reconstruction_111NumDims), this.NumDims.onnx__Mul_320);

            % Add:
            x_reconstruction_105 = x_reconstruction_110 + this.Vars.onnx__Add_321;
            x_reconstruction_105NumDims = max(coder.const(x_reconstruction_110NumDims), this.NumDims.onnx__Add_321);

            % Div:
            x_reconstruction_118 = x_reconstruction_105 ./ this.Vars.x_reconstruction_117;
            x_reconstruction_118NumDims = max(coder.const(x_reconstruction_105NumDims), this.NumDims.x_reconstruction_117);

            % Erf:
            x_reconstruction_119 = erf(x_reconstruction_118);
            x_reconstruction_119NumDims = coder.const(x_reconstruction_118NumDims);

            % Add:
            x_reconstruction_114 = x_reconstruction_119 + this.Vars.x_reconstruction_115;
            x_reconstruction_114NumDims = max(coder.const(x_reconstruction_119NumDims), this.NumDims.x_reconstruction_115);

            % Mul:
            x_reconstruction_121 = x_reconstruction_105 .* x_reconstruction_114;
            x_reconstruction_121NumDims = max(coder.const(x_reconstruction_105NumDims), coder.const(x_reconstruction_114NumDims));

            % Mul:
            x_reconstruction_120 = x_reconstruction_121 .* this.Vars.x_reconstruction_116;
            x_reconstruction_120NumDims = max(coder.const(x_reconstruction_121NumDims), this.NumDims.x_reconstruction_116);

            % Flatten:
            [dim11016, dim21017, x_reconstruction_123NumDims] = AresLidarONNX.coder.ops.prepareFlattenArgs(x_reconstruction_120, 1, coder.const(x_reconstruction_120NumDims));
            x_reconstruction_123 = reshape(x_reconstruction_120, dim11016, dim21017);

            % Gemm:
            [A1018, B1019, C1020, alpha1021, beta1022, x_reconstruction_124NumDims] = AresLidarONNX.coder.ops.prepareGemmArgs(x_reconstruction_123, this.reconstruction_mo_15, this.reconstruction_mo_14, this.Vars.Gemmalpha1007, this.Vars.Gemmbeta1008, 0, 1, this.NumDims.reconstruction_mo_14);
            x_reconstruction_124 = alpha1021*B1019*A1018 + beta1022*C1020;

            % Div:
            x_reconstruction_129 = x_reconstruction_124 ./ this.Vars.x_reconstruction_128;
            x_reconstruction_129NumDims = max(coder.const(x_reconstruction_124NumDims), this.NumDims.x_reconstruction_128);

            % Erf:
            x_reconstruction_130 = erf(x_reconstruction_129);
            x_reconstruction_130NumDims = coder.const(x_reconstruction_129NumDims);

            % Add:
            x_reconstruction_125 = x_reconstruction_130 + this.Vars.x_reconstruction_126;
            x_reconstruction_125NumDims = max(coder.const(x_reconstruction_130NumDims), this.NumDims.x_reconstruction_126);

            % Mul:
            x_reconstruction_132 = x_reconstruction_124 .* x_reconstruction_125;
            x_reconstruction_132NumDims = max(coder.const(x_reconstruction_124NumDims), coder.const(x_reconstruction_125NumDims));

            % Mul:
            x_reconstruction_131 = x_reconstruction_132 .* this.Vars.x_reconstruction_127;
            x_reconstruction_131NumDims = max(coder.const(x_reconstruction_132NumDims), this.NumDims.x_reconstruction_127);

            % LayerNormalization:
            [x_reconstruction_133, x_reconstruction_133NumDims] = AresLidarONNX.coder.ops.onnxLayerNormalization(x_reconstruction_131, this.reconstruction_mo_17, this.reconstruction_mo_16, -1, 1.000000e-05, coder.const(x_reconstruction_131NumDims));

            % Reshape:
            [shape1023, x_reconstruction__17NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(x_reconstruction_133, this.Vars.x_reconstruction_m_6, coder.const(x_reconstruction_133NumDims), 0);
            x_reconstruction__17 = reshape(x_reconstruction_133, shape1023{:});

            % Set graph output arguments
            x_reconstruction__17NumDims1009 = coder.const(x_reconstruction__17NumDims);

        end

    end

end