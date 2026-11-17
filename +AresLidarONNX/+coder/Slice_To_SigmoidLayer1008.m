classdef Slice_To_SigmoidLayer1008 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = AresLidarONNX.coder.Slice_To_SigmoidLayer1008(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = AresLidarONNX.Slice_To_SigmoidLayer1008(cgInstance.Name);
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
        function this = Slice_To_SigmoidLayer1008(mlInstance)
            this.Name = mlInstance.Name;
            this.NumOutputs = 2;
            this.OutputNames = {'range_normalized', 'validity_probability'};
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

        function [range_normalized, validity_probability] = predict(this, x_reconstruction__58__)
            if isdlarray(x_reconstruction__58__)
                x_reconstruction__58_ = stripdims(x_reconstruction__58__);
            else
                x_reconstruction__58_ = x_reconstruction__58__;
            end
            x_reconstruction__58NumDims = 4;
            x_reconstruction__58 = AresLidarONNX.coder.ops.permuteInputVar(x_reconstruction__58_, [4 3 1 2], 4);

            [range_normalized__, validity_probability__, range_normalizedNumDims__, validity_probabilityNumDims__] = Slice_To_SigmoidGraph1021(this, x_reconstruction__58, x_reconstruction__58NumDims, false);
            range_normalized_ = AresLidarONNX.coder.ops.permuteOutputVar(range_normalized__, ['as-is'], 3);
            validity_probability_ = AresLidarONNX.coder.ops.permuteOutputVar(validity_probability__, ['as-is'], 3);

            range_normalized = dlarray(single(range_normalized_), repmat('U', 1, max(2, coder.const(range_normalizedNumDims__))));
            validity_probability = dlarray(single(validity_probability_), repmat('U', 1, max(2, coder.const(validity_probabilityNumDims__))));
        end

        function [range_normalized, validity_probability, range_normalizedNumDims1022, validity_probabilityNumDims1023] = Slice_To_SigmoidGraph1021(this, x_reconstruction__58, x_reconstruction__58NumDims, Training)

            % Execute the operators:
            % Slice:
            [indices1043, x_reconstruction__21NumDims] = AresLidarONNX.coder.ops.prepareSliceArgs(x_reconstruction__58, this.Vars.x_reconstruction_m_9, this.Vars.x_reconstruction__10, this.Vars.x_reconstruction_m_8, this.Vars.x_reconstruction__11, coder.const(x_reconstruction__58NumDims));
            x_reconstruction__21 = x_reconstruction__58(indices1043{:});

            % Slice:
            [indices1044, x_reconstruction__20NumDims] = AresLidarONNX.coder.ops.prepareSliceArgs(x_reconstruction__21, this.Vars.x_reconstruction_m_1, this.Vars.x_reconstruction_m_2, this.Vars.x_reconstruction__12, this.Vars.x_reconstruction_m_3, coder.const(x_reconstruction__21NumDims));
            x_reconstruction__20 = x_reconstruction__21(indices1044{:});

            % Gather:
            [x_reconstruction__15, x_reconstruction__15NumDims] = AresLidarONNX.coder.ops.onnxGather(x_reconstruction__20, this.Vars.x_reconstruction_m_4, 1, coder.const(x_reconstruction__20NumDims), this.NumDims.x_reconstruction_m_4);

            % Sigmoid:
            X1045 = dlarray(AresLidarONNX.coder.ops.extractIfDlarray(x_reconstruction__15));
            Y1046 = sigmoid(X1045);
            validity_probability = AresLidarONNX.coder.ops.extractIfDlarray(Y1046);
            validity_probabilityNumDims = coder.const(x_reconstruction__15NumDims);

            % Gather:
            [x_reconstruction__14, x_reconstruction__14NumDims] = AresLidarONNX.coder.ops.onnxGather(x_reconstruction__20, this.Vars.x_reconstruction__13, 1, coder.const(x_reconstruction__20NumDims), this.NumDims.x_reconstruction__13);

            % Sigmoid:
            X1047 = dlarray(AresLidarONNX.coder.ops.extractIfDlarray(x_reconstruction__14));
            Y1048 = sigmoid(X1047);
            range_normalized = AresLidarONNX.coder.ops.extractIfDlarray(Y1048);
            range_normalizedNumDims = coder.const(x_reconstruction__14NumDims);

            % Set graph output arguments
            range_normalizedNumDims1022 = coder.const(range_normalizedNumDims);
            validity_probabilityNumDims1023 = coder.const(validity_probabilityNumDims);

        end

    end

end