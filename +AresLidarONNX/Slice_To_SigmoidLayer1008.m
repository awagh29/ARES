classdef Slice_To_SigmoidLayer1008 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

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
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'AresLidarONNX.coder.Slice_To_SigmoidLayer1008';
        end
    end


    methods
        function this = Slice_To_SigmoidLayer1008(name)
            this.Name = name;
            this.NumOutputs = 2;
            this.OutputNames = {'range_normalized', 'validity_probability'};
        end

        function [range_normalized, validity_probability] = predict(this, x_reconstruction__58)
            if isdlarray(x_reconstruction__58)
                x_reconstruction__58 = stripdims(x_reconstruction__58);
            end
            x_reconstruction__58NumDims = 4;
            x_reconstruction__58 = AresLidarONNX.ops.permuteInputVar(x_reconstruction__58, [4 3 1 2], 4);

            [range_normalized, validity_probability, range_normalizedNumDims, validity_probabilityNumDims] = Slice_To_SigmoidGraph1021(this, x_reconstruction__58, x_reconstruction__58NumDims, false);
            range_normalized = AresLidarONNX.ops.permuteOutputVar(range_normalized, ['as-is'], 3);
            validity_probability = AresLidarONNX.ops.permuteOutputVar(validity_probability, ['as-is'], 3);

            range_normalized = dlarray(single(range_normalized), repmat('U', 1, max(2, range_normalizedNumDims)));
            validity_probability = dlarray(single(validity_probability), repmat('U', 1, max(2, validity_probabilityNumDims)));
        end

        function [range_normalized, validity_probability] = forward(this, x_reconstruction__58)
            if isdlarray(x_reconstruction__58)
                x_reconstruction__58 = stripdims(x_reconstruction__58);
            end
            x_reconstruction__58NumDims = 4;
            x_reconstruction__58 = AresLidarONNX.ops.permuteInputVar(x_reconstruction__58, [4 3 1 2], 4);

            [range_normalized, validity_probability, range_normalizedNumDims, validity_probabilityNumDims] = Slice_To_SigmoidGraph1021(this, x_reconstruction__58, x_reconstruction__58NumDims, true);
            range_normalized = AresLidarONNX.ops.permuteOutputVar(range_normalized, ['as-is'], 3);
            validity_probability = AresLidarONNX.ops.permuteOutputVar(validity_probability, ['as-is'], 3);

            range_normalized = dlarray(single(range_normalized), repmat('U', 1, max(2, range_normalizedNumDims)));
            validity_probability = dlarray(single(validity_probability), repmat('U', 1, max(2, validity_probabilityNumDims)));
        end

        function [range_normalized, validity_probability, range_normalizedNumDims1022, validity_probabilityNumDims1023] = Slice_To_SigmoidGraph1021(this, x_reconstruction__58, x_reconstruction__58NumDims, Training)

            % Execute the operators:
            % Slice:
            [Indices, x_reconstruction__21NumDims] = AresLidarONNX.ops.prepareSliceArgs(x_reconstruction__58, this.Vars.x_reconstruction_m_9, this.Vars.x_reconstruction__10, this.Vars.x_reconstruction_m_8, this.Vars.x_reconstruction__11, x_reconstruction__58NumDims);
            x_reconstruction__21 = x_reconstruction__58(Indices{:});

            % Slice:
            [Indices, x_reconstruction__20NumDims] = AresLidarONNX.ops.prepareSliceArgs(x_reconstruction__21, this.Vars.x_reconstruction_m_1, this.Vars.x_reconstruction_m_2, this.Vars.x_reconstruction__12, this.Vars.x_reconstruction_m_3, x_reconstruction__21NumDims);
            x_reconstruction__20 = x_reconstruction__21(Indices{:});

            % Gather:
            [x_reconstruction__15, x_reconstruction__15NumDims] = AresLidarONNX.ops.onnxGather(x_reconstruction__20, this.Vars.x_reconstruction_m_4, 1, x_reconstruction__20NumDims, this.NumDims.x_reconstruction_m_4);

            % Sigmoid:
            validity_probability = sigmoid(dlarray(x_reconstruction__15));
            validity_probabilityNumDims = x_reconstruction__15NumDims;

            % Gather:
            [x_reconstruction__14, x_reconstruction__14NumDims] = AresLidarONNX.ops.onnxGather(x_reconstruction__20, this.Vars.x_reconstruction__13, 1, x_reconstruction__20NumDims, this.NumDims.x_reconstruction__13);

            % Sigmoid:
            range_normalized = sigmoid(dlarray(x_reconstruction__14));
            range_normalizedNumDims = x_reconstruction__14NumDims;

            % Set graph output arguments
            range_normalizedNumDims1022 = range_normalizedNumDims;
            validity_probabilityNumDims1023 = validity_probabilityNumDims;

        end

    end

end