function [struc] = driverTOPT(mesh_file)

% Limit on volume
VolumeFraction = 0.3;
% Density model
model_t = 'simp';

% Domain specifications
Domain.xmin = 0;  % min dim in x-dir
Domain.xmax = 1.6;   % max dim in x-dir
Domain.ymin = 0;  % min dim in y-dir
Domain.ymax = 1;   % max dim in y-dir
Domain.nx = 64;     % num intervals in x-dir
Domain.ny = 40;     % num intervals in y-dir
% regularization parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV'; 
beta = 2*(Domain.xmax / (Domain.nx)); 
gamma = 1e-1;

% generate problem-specific constant quantities
rhs_fn=@(usr_par)generateNodalForce(usr_par);
[struc] = generateParams(Domain, rhs_fn);

% store problem-specific constant quantites
struc.reg = reg;
struc.beta = beta;
struc.gamma = gamma;
struc.VolumeFraction = VolumeFraction;
struc.model_t = model_t;
% Normalization factors
struc.theta = 1;
struc.alpha = 1;
% problem parameters
struc.SimpPenalty = 3;

end