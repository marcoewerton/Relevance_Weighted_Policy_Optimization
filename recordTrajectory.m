clear variables;
close;
clc;

n_trajectories = 5;
trajectories = cell(n_trajectories,1);

for i = 1:n_trajectories
    
    close;
    figure(i);
    hold on;
    grid on;
    axis([0 1 0 1]);
    plot([0.5 0.5], [0 0.45], 'b', 'LineWidth', 5);
    plot([0.5 0.5], [0.55 1], 'b', 'LineWidth', 5);
    plot(0.1, 0.1, 'gx', 'MarkerSize', 25, 'LineWidth', 10);
    plot(0.9, 0.9, 'rx', 'MarkerSize', 25, 'LineWidth', 10);
    
    h = imfreehand('Closed', false);
    
    % get the position (x,y coordinates) of each point of the curve
    positions = getPosition(h);

    trajectories{i,1} = positions;
    
end

% save('trajectories.mat', 'trajectories');
% save('trajectories2.mat', 'trajectories');
% save('trajectories3.mat', 'trajectories');
% save('trajectories4.mat', 'trajectories');
% save('trajectories5.mat', 'trajectories');
%save('trajectories6.mat', 'trajectories');