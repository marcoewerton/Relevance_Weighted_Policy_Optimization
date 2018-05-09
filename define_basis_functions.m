% define basis functions

function PSIs_matrix = define_basis_functions(z_T, N, alpha)

    % z_T: phase function z evaluated at the last time step
    % N: number of basis functions
        
    h = z_T;
    
    %% Define generic basis function
    psi = @(z_t,n) exp(-0.5*(z_t-(n-1)*z_T/(N-1)).^2/h);
    
    %% Define PSIs_matrix
    z = alpha:alpha:z_T;
    phase = repmat(z', 1, N);
    n = repmat(1:N, size(z,2), 1);
    PSIs_matrix = psi(phase, n);
    normalizer = sum(PSIs_matrix, 2);
    PSIs_matrix = bsxfun(@rdivide, PSIs_matrix, normalizer);

end