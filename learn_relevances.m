clear variables;
close;
clc;

tic

light_gray = [0.7 0.7 0.7];

%% load trajectories that have been previously recorded
%load('trajectories.mat');
%load('trajectories2.mat');
%load('trajectories3.mat');
load('trajectories4.mat');
%load('trajectories5.mat');
%load('trajectories6.mat');
n_trajectories = length(trajectories);

%% define number of basis functions
N = 10;

%% time-align trajectories
t_align_trajectories = trajectories;
max_n_time_steps = 0; % variable to record the maximum number of time steps
for i = 1:n_trajectories
    if size(trajectories{i}, 1) > max_n_time_steps
       max_n_time_steps = size(trajectories{i}, 1); 
    end
end

xq = linspace(0,1,max_n_time_steps);

for i = 1:n_trajectories
    x = linspace(0,1,size(trajectories{i}, 1));
    t_align_trajectories{i} = interp1(x,trajectories{i},xq);
end

%% define basis functions
PSIs_matrix = define_basis_functions(max_n_time_steps, N, 1);

%% determine block matrix of PSIs_matrices
blk_PSI = blkdiag(PSIs_matrix, PSIs_matrix);

%% compute ProMP
promp = promp(t_align_trajectories, PSIs_matrix);

%% sample from the ProMP
n_samples = 100;
weights_samples_from_promp = mvnrnd(promp.mu', promp.Sigma, n_samples);
trajectory_samples_from_promp = weights_samples_from_promp*blk_PSI';

%% plot samples from the ProMP
close;
figure(1);
hold on;
grid on;
axis([0 1 0 1]);
plot([0.5 0.5], [0 0.45], 'b', 'LineWidth', 5);
plot([0.5 0.5], [0.55 1], 'b', 'LineWidth', 5);
plot(0.1, 0.1, 'gx', 'MarkerSize', 25, 'LineWidth', 10);
plot(0.9, 0.9, 'rx', 'MarkerSize', 25, 'LineWidth', 10);
for i = 1:n_samples
    plot(trajectory_samples_from_promp(i,1:max_n_time_steps), trajectory_samples_from_promp(i,max_n_time_steps+1:end), 'LineWidth', 2, 'Color', light_gray);
end

start_pos = [0.1, 0.1];
end_pos = [0.9, 0.9];
center = [0.5, 0.5];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% learn relevance functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_iterations_rel_opt = 50;
n_samples_rel_opt = 200;
n_trajectory_samples_rel_opt = 50;

% hyperparameters of the relevance functions
k = 1;
n_r = 3;
m1 = N/n_r;
m2 = 2*N/n_r;

% hyperparameters of the RWR for improving the relevance functions
beta_rel_opt = 10;

x = 1:N;
rho1 = 1./(1+exp(-k*(x-m1)));
rho2 = 1./(1+exp(-k*(x-m2)));
rho3 = ones(1,N);
P = [rho1; rho2; rho3];

std_start_distances_rel_opt_1 = NaN(n_samples_rel_opt, n_iterations_rel_opt);
std_end_distances_rel_opt_1 = NaN(n_samples_rel_opt, n_iterations_rel_opt);
std_center_distances_rel_opt_1 = NaN(n_samples_rel_opt, n_iterations_rel_opt);

std_start_distances_rel_opt_2 = NaN(n_samples_rel_opt, n_iterations_rel_opt);
std_end_distances_rel_opt_2 = NaN(n_samples_rel_opt, n_iterations_rel_opt);
std_center_distances_rel_opt_2 = NaN(n_samples_rel_opt, n_iterations_rel_opt);

std_start_distances_rel_opt_3 = NaN(n_samples_rel_opt, n_iterations_rel_opt);
std_end_distances_rel_opt_3 = NaN(n_samples_rel_opt, n_iterations_rel_opt);
std_center_distances_rel_opt_3 = NaN(n_samples_rel_opt, n_iterations_rel_opt);

returns1 = NaN(n_samples_rel_opt, n_iterations_rel_opt);
returns2 = NaN(n_samples_rel_opt, n_iterations_rel_opt);
returns3 = NaN(n_samples_rel_opt, n_iterations_rel_opt);

mu_r = zeros(1,3*3); % there is one parameter for each rho and for each aspect of the reward
Sigma_r = eye(3*3);

new_mu_r = mu_r*0;
new_Sigma_r = Sigma_r*0;

for rel_opt_iter = 1:n_iterations_rel_opt
    fprintf(['Relevance Optimization Iteration ', int2str(rel_opt_iter), '\n'])
    
    r_samples = mvnrnd(mu_r, Sigma_r, n_samples_rel_opt); % sample coefficients of basis functions for the relevance
    
    % define relevances based on the samples
    rel_2_start = r_samples(:,1:3)*P;
    rel_2_start = max(rel_2_start, 0);
    rel_2_start = min(rel_2_start, 1);
        
    rel_2_center = r_samples(:,4:6)*P;
    rel_2_center = max(rel_2_center, 0);
    rel_2_center = min(rel_2_center, 1);
        
    rel_2_end = r_samples(:,7:9)*P;
    rel_2_end = max(rel_2_end, 0);
    rel_2_end = min(rel_2_end, 1);
        
    for i = 1:n_samples_rel_opt
        weights_cov_with_rel_2_start = zeros(size(promp.Sigma, 1));
        weights_cov_with_rel_2_center = zeros(size(promp.Sigma, 1));
        weights_cov_with_rel_2_end = zeros(size(promp.Sigma, 1));
        
        weights_cov_with_rel_2_start(1:1+size(weights_cov_with_rel_2_start,1):end) = diag(promp.Sigma)'.*[rel_2_start(i,:), rel_2_start(i,:)];
        weights_cov_with_rel_2_center(1:1+size(weights_cov_with_rel_2_center,1):end) = diag(promp.Sigma)'.*[rel_2_center(i,:), rel_2_center(i,:)];
        weights_cov_with_rel_2_end(1:1+size(weights_cov_with_rel_2_end,1):end) = diag(promp.Sigma)'.*[rel_2_end(i,:), rel_2_end(i,:)];
        
        weights_samples_from_promp_wr2s = mvnrnd(promp.mu', weights_cov_with_rel_2_start, n_trajectory_samples_rel_opt); % n_trajectory_samples_rel_opt X 2*N;   wr2s = with relevance to start
        weights_samples_from_promp_wr2c = mvnrnd(promp.mu', weights_cov_with_rel_2_center, n_trajectory_samples_rel_opt);
        weights_samples_from_promp_wr2e = mvnrnd(promp.mu', weights_cov_with_rel_2_end, n_trajectory_samples_rel_opt);
        
        trajectory_samples_from_promp_wr2s = weights_samples_from_promp_wr2s*blk_PSI'; % n_trajectory_samples_rel_opt X 2*max_n_time_steps
        trajectory_samples_from_promp_wr2c = weights_samples_from_promp_wr2c*blk_PSI';
        trajectory_samples_from_promp_wr2e = weights_samples_from_promp_wr2e*blk_PSI';
        
        start_dist_1 = bsxfun(@minus, trajectory_samples_from_promp_wr2s(:,[1, max_n_time_steps+1]), start_pos);
        start_dist_1 = sqrt(sum(start_dist_1.^2, 2));
        
        start_dist_2 = bsxfun(@minus, trajectory_samples_from_promp_wr2c(:,[1, max_n_time_steps+1]), start_pos);
        start_dist_2 = sqrt(sum(start_dist_2.^2, 2));
        
        start_dist_3 = bsxfun(@minus, trajectory_samples_from_promp_wr2e(:,[1, max_n_time_steps+1]), start_pos);
        start_dist_3 = sqrt(sum(start_dist_3.^2, 2));
        
        all_center_dists_in_current_iteration_1 = NaN(n_trajectory_samples_rel_opt, max_n_time_steps);
        all_center_dists_in_current_iteration_2 = NaN(n_trajectory_samples_rel_opt, max_n_time_steps);
        all_center_dists_in_current_iteration_3 = NaN(n_trajectory_samples_rel_opt, max_n_time_steps);
        % compute distance to the center
        for t = 1:max_n_time_steps
            center_dist_1 = bsxfun(@minus, trajectory_samples_from_promp_wr2s(:,[t, t+max_n_time_steps]), center);
            center_dist_1 = sqrt(sum(center_dist_1.^2, 2));
            all_center_dists_in_current_iteration_1(:,t) = center_dist_1;
            
            center_dist_2 = bsxfun(@minus, trajectory_samples_from_promp_wr2c(:,[t, t+max_n_time_steps]), center);
            center_dist_2 = sqrt(sum(center_dist_2.^2, 2));
            all_center_dists_in_current_iteration_2(:,t) = center_dist_2;
            
            center_dist_3 = bsxfun(@minus, trajectory_samples_from_promp_wr2e(:,[t, t+max_n_time_steps]), center);
            center_dist_3 = sqrt(sum(center_dist_3.^2, 2));
            all_center_dists_in_current_iteration_3(:,t) = center_dist_3;
        end
        center_dist_1 = min(all_center_dists_in_current_iteration_1, [], 2);
        center_dist_2 = min(all_center_dists_in_current_iteration_2, [], 2);
        center_dist_3 = min(all_center_dists_in_current_iteration_3, [], 2);
        
        end_dist_1 = bsxfun(@minus, trajectory_samples_from_promp_wr2s(:,[max_n_time_steps, end]), end_pos);
        end_dist_1 = sqrt(sum(end_dist_1.^2, 2));
        
        end_dist_2 = bsxfun(@minus, trajectory_samples_from_promp_wr2c(:,[max_n_time_steps, end]), end_pos);
        end_dist_2 = sqrt(sum(end_dist_2.^2, 2));
        
        end_dist_3 = bsxfun(@minus, trajectory_samples_from_promp_wr2e(:,[max_n_time_steps, end]), end_pos);
        end_dist_3 = sqrt(sum(end_dist_3.^2, 2));
        
        std_start_distances_rel_opt_1(i, rel_opt_iter) = std(start_dist_1); % these standard deviations work as rewards or costs
        std_center_distances_rel_opt_1(i, rel_opt_iter) = std(center_dist_1); % these standard deviations work as rewards or costs
        std_end_distances_rel_opt_1(i, rel_opt_iter) = std(end_dist_1); % these standard deviations work as rewards or costs
        
        std_start_distances_rel_opt_2(i, rel_opt_iter) = std(start_dist_2); % these standard deviations work as rewards or costs
        std_center_distances_rel_opt_2(i, rel_opt_iter) = std(center_dist_2); % these standard deviations work as rewards or costs
        std_end_distances_rel_opt_2(i, rel_opt_iter) = std(end_dist_2); % these standard deviations work as rewards or costs
        
        std_start_distances_rel_opt_3(i, rel_opt_iter) = std(start_dist_3); % these standard deviations work as rewards or costs
        std_center_distances_rel_opt_3(i, rel_opt_iter) = std(center_dist_3); % these standard deviations work as rewards or costs
        std_end_distances_rel_opt_3(i, rel_opt_iter) = std(end_dist_3); % these standard deviations work as rewards or costs
    end
    
    R_1 = + std_start_distances_rel_opt_1(:,rel_opt_iter) - std_center_distances_rel_opt_1(:, rel_opt_iter) - std_end_distances_rel_opt_1(:, rel_opt_iter);
    R_2 = - std_start_distances_rel_opt_2(:,rel_opt_iter) + std_center_distances_rel_opt_2(:, rel_opt_iter) - std_end_distances_rel_opt_2(:, rel_opt_iter);
    R_3 = - std_start_distances_rel_opt_3(:,rel_opt_iter) - std_center_distances_rel_opt_3(:, rel_opt_iter) + std_end_distances_rel_opt_3(:, rel_opt_iter);
    
    reward_weight_1 = exp(beta_rel_opt*R_1);
    reward_weight_2 = exp(beta_rel_opt*R_2);
    reward_weight_3 = exp(beta_rel_opt*R_3);
    
    returns1(:,rel_opt_iter) = reward_weight_1;
    returns2(:,rel_opt_iter) = reward_weight_2;
    returns3(:,rel_opt_iter) = reward_weight_3;
        
    new_mu_r(1,1:3) = reward_weight_1'*r_samples(:,1:3);
    new_mu_r(1,4:6) = reward_weight_2'*r_samples(:,4:6);
    new_mu_r(1,7:9) = reward_weight_3'*r_samples(:,7:9);
    
    W_minus_mu_1 = bsxfun(@minus, r_samples(:,1:3), mu_r(1,1:3));
    W_minus_mu_2 = bsxfun(@minus, r_samples(:,4:6), mu_r(1,4:6));
    W_minus_mu_3 = bsxfun(@minus, r_samples(:,7:9), mu_r(1,7:9));
    
    new_Sigma_r(1:3,1:3) = new_Sigma_r(1:3,1:3)*0;
    new_Sigma_r(4:6,4:6) = new_Sigma_r(4:6,4:6)*0;
    new_Sigma_r(7:9,7:9) = new_Sigma_r(7:9,7:9)*0;
    
    for i = 1:n_samples_rel_opt
        new_Sigma_r(1:3,1:3) = new_Sigma_r(1:3,1:3) + reward_weight_1(i)*W_minus_mu_1(i,:)'*W_minus_mu_1(i,:);
        new_Sigma_r(4:6,4:6) = new_Sigma_r(4:6,4:6) + reward_weight_2(i)*W_minus_mu_2(i,:)'*W_minus_mu_2(i,:);
        new_Sigma_r(7:9,7:9) = new_Sigma_r(7:9,7:9) + reward_weight_3(i)*W_minus_mu_3(i,:)'*W_minus_mu_3(i,:);
    end
    
    new_mu_r(1,1:3) = new_mu_r(1,1:3)/sum(reward_weight_1);
    new_mu_r(1,4:6) = new_mu_r(1,4:6)/sum(reward_weight_2);
    new_mu_r(1,7:9) = new_mu_r(1,7:9)/sum(reward_weight_3);    
    
    new_Sigma_r(1:3,1:3) = new_Sigma_r(1:3,1:3)/sum(reward_weight_1);
    new_Sigma_r(4:6,4:6) = new_Sigma_r(4:6,4:6)/sum(reward_weight_2);
    new_Sigma_r(7:9,7:9) = new_Sigma_r(7:9,7:9)/sum(reward_weight_3);
    
    mu_r(1,1:3) = new_mu_r(1,1:3);
    mu_r(1,4:6) = new_mu_r(1,4:6);
    mu_r(1,7:9) = new_mu_r(1,7:9);
    
    Sigma_r = new_Sigma_r;
    Sigma_r = (Sigma_r + Sigma_r')/2;
        
end

relevance_to_start = mean(rel_2_start);
relevance_to_start = relevance_to_start/max(relevance_to_start);
relevance_to_center = mean(rel_2_center);
relevance_to_center = relevance_to_center/max(relevance_to_center);
relevance_to_end = mean(rel_2_end);
relevance_to_end = relevance_to_end/max(relevance_to_end);

relevance = [relevance_to_start; relevance_to_center; relevance_to_end];

%%%%%%%%%%%
figure(2);
hold on;
axis([1, N, 0, 1]);
plot(1:N, relevance_to_start, 'r', 'LineWidth', 2);
plot(1:N, relevance_to_center, 'g', 'LineWidth', 2);
plot(1:N, relevance_to_end, 'b', 'LineWidth', 2);
xlabel('weight index');
ylabel('relevance');
legend('start', 'center', 'end');
%%%%%%%%%%%

save('relevance.mat', 'relevance');

save('workspace_after_relevance_learning.mat');

toc

%% compute mean ans std of returns1
mu_returns1 = mean(returns1);
std_returns1 = std(returns1);
figure(12);
hold on;
grid on;
shadedErrorBar(1:n_iterations_rel_opt, mu_returns1, 2*std_returns1, '-b', 2);
xlabel('Iteration');
ylabel('returns1');

%% compute mean ans std of returns2
mu_returns2 = mean(returns2);
std_returns2 = std(returns2);
figure(13);
hold on;
grid on;
shadedErrorBar(1:n_iterations_rel_opt, mu_returns2, 2*std_returns2, '-b', 2);
xlabel('Iteration');
ylabel('returns2');

%% compute mean ans std of returns3
mu_returns3 = mean(returns3);
std_returns3 = std(returns3);
figure(14);
hold on;
grid on;
shadedErrorBar(1:n_iterations_rel_opt, mu_returns3, 2*std_returns3, '-b', 2);
xlabel('Iteration');
ylabel('returns3');
