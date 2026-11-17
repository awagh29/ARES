classdef ReshapeLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = AresLidarONNX.coder.ReshapeLayer1000(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = AresLidarONNX.ReshapeLayer1000(cgInstance.Name);
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
        function this = ReshapeLayer1000(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_reconstruction__19'};
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

        function [x_reconstruction__19] = predict(this, lidar_history__)
            if isdlarray(lidar_history__)
                lidar_history_ = stripdims(lidar_history__);
            else
                lidar_history_ = lidar_history__;
            end
            lidar_historyNumDims = 5;
            lidar_history = AresLidarONNX.coder.ops.permuteInputVar(lidar_history_, ['as-is'], 5);

            [x_reconstruction__19__, x_reconstruction__19NumDims__] = ReshapeGraph1000(this, lidar_history, lidar_historyNumDims, false);
            x_reconstruction__19_ = AresLidarONNX.coder.ops.permuteOutputVar(x_reconstruction__19__, [3 4 2 1], 4);

            x_reconstruction__19 = dlarray(single(x_reconstruction__19_), 'SSCB');
        end

        function [x_reconstruction__19, x_reconstruction__19NumDims1001] = ReshapeGraph1000(this, lidar_history, lidar_historyNumDims, Training)

            % Execute the operators:
            % Reshape:
            [shape1000, x_reconstruction__19NumDims] = AresLidarONNX.coder.ops.prepareReshapeArgs(lidar_history, this.Vars.x_reconstruction_m_5, coder.const(lidar_historyNumDims), 0);
            x_reconstruction__19 = reshape(lidar_history, shape1000{:});

            % Set graph output arguments
            x_reconstruction__19NumDims1001 = coder.const(x_reconstruction__19NumDims);

        end

    end

end