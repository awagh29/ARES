classdef ReshapeLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'AresLidarONNX.coder.ReshapeLayer1000';
        end
    end


    methods
        function this = ReshapeLayer1000(name)
            this.Name = name;
            this.OutputNames = {'x_reconstruction__19'};
        end

        function [x_reconstruction__19] = predict(this, lidar_history)
            if isdlarray(lidar_history)
                lidar_history = stripdims(lidar_history);
            end
            lidar_historyNumDims = 5;
            lidar_history = AresLidarONNX.ops.permuteInputVar(lidar_history, ['as-is'], 5);

            [x_reconstruction__19, x_reconstruction__19NumDims] = ReshapeGraph1000(this, lidar_history, lidar_historyNumDims, false);
            x_reconstruction__19 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__19, [3 4 2 1], 4);

            x_reconstruction__19 = dlarray(single(x_reconstruction__19), 'SSCB');
        end

        function [x_reconstruction__19] = forward(this, lidar_history)
            if isdlarray(lidar_history)
                lidar_history = stripdims(lidar_history);
            end
            lidar_historyNumDims = 5;
            lidar_history = AresLidarONNX.ops.permuteInputVar(lidar_history, ['as-is'], 5);

            [x_reconstruction__19, x_reconstruction__19NumDims] = ReshapeGraph1000(this, lidar_history, lidar_historyNumDims, true);
            x_reconstruction__19 = AresLidarONNX.ops.permuteOutputVar(x_reconstruction__19, [3 4 2 1], 4);

            x_reconstruction__19 = dlarray(single(x_reconstruction__19), 'SSCB');
        end

        function [x_reconstruction__19, x_reconstruction__19NumDims1001] = ReshapeGraph1000(this, lidar_history, lidar_historyNumDims, Training)

            % Execute the operators:
            % Reshape:
            [shape, x_reconstruction__19NumDims] = AresLidarONNX.ops.prepareReshapeArgs(lidar_history, this.Vars.x_reconstruction_m_5, lidar_historyNumDims, 0);
            x_reconstruction__19 = reshape(lidar_history, shape{:});

            % Set graph output arguments
            x_reconstruction__19NumDims1001 = x_reconstruction__19NumDims;

        end

    end

end