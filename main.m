% main script to test RL with ProMPs

clear variables;
close;
clc;

tic

wall_width = 0.2;

dt = 0.01;

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

close;
figure(10);
hold on;
grid on;
set_fig_position([0.349 0.339 0.161 0.262]);
axis([0 1 0 1]);
plot([0.5 0.5], [0 0.45], 'b', 'LineWidth', 5);
plot([0.5 0.5], [0.55 1], 'b', 'LineWidth', 5);
plot(0.1, 0.1, 'gx', 'MarkerSize', 15, 'LineWidth', 5);
plot(0.9, 0.9, 'rx', 'MarkerSize', 15, 'LineWidth', 5);
for i = 1:length(t_align_trajectories)
   plot(t_align_trajectories{i}(:,1), t_align_trajectories{i}(:,2), 'LineWidth', 2, 'Color', light_gray); 
end

%% define basis functions
PSIs_matrix = define_basis_functions(max_n_time_steps, N, 1);

%% determine block matrix of PSIs_matrices
blk_PSI = blkdiag(PSIs_matrix, PSIs_matrix);

%% compute ProMP
promp = promp(t_align_trajectories, PSIs_matrix);

%% sample from the ProMP
n_samples = 1000;
weights_samples_from_promp = mvnrnd(promp.mu', promp.Sigma, n_samples);
trajectory_samples_from_promp = weights_samples_from_promp*blk_PSI';

%% plot samples from the ProMP
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

%% Reward-Weighted Regression
start_pos = [0.1, 0.1];
end_pos = [0.9, 0.9];
center = [0.5, 0.5];

beta = 20;

new_promp_mu = promp.mu*0;
new_promp_Sigma = promp.Sigma*0;


n_iterations = 100;
returns = NaN(n_samples, n_iterations);

start_distances = NaN(n_samples, n_iterations);
end_distances = NaN(n_samples, n_iterations);
center_distances = NaN(n_samples, n_iterations);
jerks = NaN(n_samples, n_iterations);
accelerations = NaN(n_samples, n_iterations);

%%
load('relevance.mat');

relevance_to_start = relevance(1,:);
relevance_to_center = relevance(2,:);
relevance_to_end = relevance(3,:);
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iteration = 1:n_iterations
    
    previous_promp_Sigma = promp.Sigma;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START
    weights_cov_with_rel = zeros(size(promp.Sigma, 1));    
    
    % rescale variance of weights with relevance
    weights_cov_with_rel(1:1+size(weights_cov_with_rel,1):end) = diag(promp.Sigma)'.*[relevance_to_start, relevance_to_start];
     
    weights_samples_from_promp = mvnrnd(promp.mu', weights_cov_with_rel, n_samples); % n_samples X 2*N
    trajectory_samples_from_promp = weights_samples_from_promp*blk_PSI'; % n_samples X 2*max_n_time_steps
    
    start_dist = bsxfun(@minus, trajectory_samples_from_promp(:,[1, max_n_time_steps+1]), start_pos);
    start_dist = sqrt(sum(start_dist.^2, 2));
    
    R = -start_dist;
    
    reward_weight = exp(beta*R);
    
    new_promp_mu = weights_samples_from_promp'*reward_weight;
    
    W_minus_mu = bsxfun(@minus, weights_samples_from_promp, promp.mu');
    
    new_promp_Sigma = new_promp_Sigma*0;
    
    for i = 1:n_samples
        new_promp_Sigma = new_promp_Sigma + reward_weight(i)*W_minus_mu(i,:)'*W_minus_mu(i,:);
    end
    
    new_promp_mu = new_promp_mu/sum(reward_weight);
    new_promp_Sigma = new_promp_Sigma/sum(reward_weight);
    
    promp.mu = new_promp_mu;
    promp.Sigma = new_promp_Sigma;
    promp.Sigma = (promp.Sigma + promp.Sigma')/2;
   
    updated_promp_Sigma = zeros(size(promp.Sigma, 1));
    updated_promp_Sigma(1:1+size(weights_cov_with_rel,1):end) = [1-relevance_to_start, 1-relevance_to_start].*diag(previous_promp_Sigma)' + [relevance_to_start, relevance_to_start].*diag(promp.Sigma)';
    promp.Sigma = updated_promp_Sigma;
    
    previous_promp_Sigma = promp.Sigma;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CENTER
    weights_cov_with_rel = zeros(size(promp.Sigma, 1));
        
    % rescale variance of weights with relevance
    weights_cov_with_rel(1:1+size(weights_cov_with_rel,1):end) = diag(promp.Sigma)'.*[relevance_to_center, relevance_to_center];
     
    weights_samples_from_promp = mvnrnd(promp.mu', weights_cov_with_rel, n_samples); % n_samples X 2*N
    trajectory_samples_from_promp = weights_samples_from_promp*blk_PSI'; % n_samples X 2*max_n_time_steps
    
    all_center_dists_in_current_iteration = NaN(n_samples, max_n_time_steps);
    
    % compute distance to the center
    for t = 1:max_n_time_steps
        center_dist = bsxfun(@minus, trajectory_samples_from_promp(:,[t, t+max_n_time_steps]), center);
        center_dist = sqrt(sum(center_dist.^2, 2));
        all_center_dists_in_current_iteration(:,t) = center_dist;
    end
    center_dist = min(all_center_dists_in_current_iteration, [], 2);
    
    R = -center_dist;
    
    reward_weight = exp(beta*R);
    
    new_promp_mu = weights_samples_from_promp'*reward_weight;
    
    W_minus_mu = bsxfun(@minus, weights_samples_from_promp, promp.mu');
    
    new_promp_Sigma = new_promp_Sigma*0;
    
    for i = 1:n_samples
        new_promp_Sigma = new_promp_Sigma + reward_weight(i)*W_minus_mu(i,:)'*W_minus_mu(i,:);
    end
    
    new_promp_mu = new_promp_mu/sum(reward_weight);
    new_promp_Sigma = new_promp_Sigma/sum(reward_weight);
    
    promp.mu = new_promp_mu;
    promp.Sigma = new_promp_Sigma;
    promp.Sigma = (promp.Sigma + promp.Sigma')/2;
    
    updated_promp_Sigma = zeros(size(promp.Sigma, 1));
    updated_promp_Sigma(1:1+size(weights_cov_with_rel,1):end) = [1-relevance_to_center, 1-relevance_to_center].*diag(previous_promp_Sigma)' + [relevance_to_center, relevance_to_center].*diag(promp.Sigma)';
    promp.Sigma = updated_promp_Sigma;
    
    previous_promp_Sigma = promp.Sigma;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END
    weights_cov_with_rel = zeros(size(promp.Sigma, 1));
        
    % rescale variance of weights with relevance
    weights_cov_with_rel(1:1+size(weights_cov_with_rel,1):end) = diag(promp.Sigma)'.*[relevance_to_end, relevance_to_end];
        
    weights_samples_from_promp = mvnrnd(promp.mu', weights_cov_with_rel, n_samples); % n_samples X 2*N
    trajectory_samples_from_promp = weights_samples_from_promp*blk_PSI'; % n_samples X 2*max_n_time_steps
    
    start_dist = bsxfun(@minus, trajectory_samples_from_promp(:,[1, max_n_time_steps+1]), start_pos);
    start_dist = sqrt(sum(start_dist.^2, 2));
    
    end_dist = bsxfun(@minus, trajectory_samples_from_promp(:,[max_n_time_steps, end]), end_pos);
    end_dist = sqrt(sum(end_dist.^2, 2));
    
    all_center_dists_in_current_iteration = NaN(n_samples, max_n_time_steps);
    
    % compute distance to the center
    for t = 1:max_n_time_steps
        center_dist = bsxfun(@minus, trajectory_samples_from_promp(:,[t, t+max_n_time_steps]), center);
        center_dist = sqrt(sum(center_dist.^2, 2));
        all_center_dists_in_current_iteration(:,t) = center_dist;
    end
    center_dist = min(all_center_dists_in_current_iteration, [], 2);
    
    R = - end_dist;
    
    reward_weight = exp(beta*R);
    
    returns(:,iteration) = exp(-beta*(start_dist+end_dist+center_dist));
    start_distances(:, iteration) = start_dist;
    end_distances(:, iteration) = end_dist;
    center_distances(:, iteration) = center_dist;
    
    new_promp_mu = weights_samples_from_promp'*reward_weight;
    
    W_minus_mu = bsxfun(@minus, weights_samples_from_promp, promp.mu');
    
    new_promp_Sigma = new_promp_Sigma*0;
    
    for i = 1:n_samples
        new_promp_Sigma = new_promp_Sigma + reward_weight(i)*W_minus_mu(i,:)'*W_minus_mu(i,:);
    end
    
    new_promp_mu = new_promp_mu/sum(reward_weight);
    new_promp_Sigma = new_promp_Sigma/sum(reward_weight);
    
    promp.mu = new_promp_mu;
    promp.Sigma = new_promp_Sigma;
    promp.Sigma = (promp.Sigma + promp.Sigma')/2;
    
    updated_promp_Sigma = zeros(size(promp.Sigma, 1));
    updated_promp_Sigma(1:1+size(weights_cov_with_rel,1):end) = [1-relevance_to_end, 1-relevance_to_end].*diag(previous_promp_Sigma)' + [relevance_to_end, relevance_to_end].*diag(promp.Sigma)';
    promp.Sigma = updated_promp_Sigma;
    
end

%% compute mean ans std of the returns
mu_returns = mean(returns);
std_returns = std(returns);
figure(3);
hold on;
grid on;
shadedErrorBar(1:n_iterations, mu_returns, 2*std_returns, '-b', 2);
xlabel('Iteration');
ylabel('Return');

%% compute mean ans std of start_dist
mu_start_dist = mean(start_distances);
std_start_dist = std(start_distances);
figure(4);
hold on;
grid on;
shadedErrorBar(1:n_iterations, mu_start_dist, 2*std_start_dist, '-b', 2);
xlabel('Iteration');
ylabel('Start distance');

%% compute mean ans std of end_dist
mu_end_dist = mean(end_distances);
std_end_dist = std(end_distances);
figure(5);
hold on;
grid on;
shadedErrorBar(1:n_iterations, mu_end_dist, 2*std_end_dist, '-b', 2);
xlabel('Iteration');
ylabel('End distance');

%% compute mean ans std of center_dist
mu_center_dist = mean(center_distances);
std_center_dist = std(center_distances);
figure(6);
hold on;
grid on;
shadedErrorBar(1:n_iterations, mu_center_dist, 2*std_center_dist, '-b', 2);
xlabel('Iteration');
ylabel('Center distance');

%% sample from the ProMP
weights_samples_from_promp = mvnrnd(promp.mu', promp.Sigma, n_samples);
trajectory_samples_from_promp = weights_samples_from_promp*blk_PSI';

%% plot samples from the ProMP
figure(2);
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

toc
