function promp = promp(t_align_trajectories, PSIs_matrix)
    % Arguments:
    % trajectories -- cell array of size (n_trajectories X 1) containing time-aligned trajectories 
    % Returns:
    % promp -- struct with mean and covariance of the ProMP
    
    n_trajectories = length(t_align_trajectories);
    n_dofs = size(t_align_trajectories{1},2);
    N = size(PSIs_matrix, 2); % number of basis functions    
    
    promp = struct;
    
    %% learn weights
    weights = learn_weights(t_align_trajectories, PSIs_matrix);
    
    %% transform cell array of weights into matrix of weights
    %% the size of this matrix of weights is (n_trajectories X (N*n_dofs))
    weights_matrix = NaN(n_trajectories, N*n_dofs);
    for i = 1:n_trajectories
        weights_matrix(i,:) = [weights{i}(:,1)' weights{i}(:,2)'];
    end
        
    mu_w = mean(weights_matrix);
    Sigma_w = cov(weights_matrix);
    
    % It's computationally better to keep working in the weight space
    promp.mu = mu_w';
    promp.Sigma = Sigma_w;
    promp.Sigma = (promp.Sigma + promp.Sigma')/2; % just to make sure promp.Sigma is symmetric
    
end

