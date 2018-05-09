%%% Learn weights for Gaussian basis functions from trajectories
% ATTENTION: Using time-aligned trajectories to generate weights!

function weights = learn_weights(t_align_trajectories, PSIs_matrix)

    % There is a matrix (#basis functions X #DoFs) of weights for each demonstration.
    weights = cell(numel(t_align_trajectories),1);

    for index = 1:numel(t_align_trajectories)
        weights{index} = (PSIs_matrix'*PSIs_matrix + 10^-12*eye(size(PSIs_matrix'*PSIs_matrix,1)))\PSIs_matrix'*t_align_trajectories{index};
    end

end