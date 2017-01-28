function [struc] = driverTOPT(mesh_file,multi_material)

% Density model
model_t = 'simp';

% regularization parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV'; 
beta = 1e-5; 
gamma = 1e-5;

% generate problem-specific constant quantities
input.neumann = false;
input.mesh_file = mesh_file;
input.multi_material = multi_material;
rhs_fn=@(struc)generateNodalForce(struc);
[struc] = generateParams(input, rhs_fn);

% store problem-specific constant quantites
struc.reg = reg;
struc.beta = beta;
struc.gamma = gamma;
struc.model_t = model_t;
% Normalization factors
struc.theta = 1;
struc.alpha = 1;
% problem parameters
struc.SimpPenalty = [3; 3];

end