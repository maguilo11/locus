function [struc] = driverTOPT(mesh_file)

% Limit on volume
VolumeFraction = 0.5;

% perimeter control parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV'; 
beta = 1e-3; 
gamma = 1e-3;

% generate problem-specific constant quantities
input.neumann = false;
input.mesh_file = mesh_file;
rhs_fn=@(struc)generateNodalForce(struc);
[struc] = generateParams(input, rhs_fn);

% store problem-specific constant quantites
struc.reg = reg;
struc.beta = beta;
struc.gamma = gamma;
struc.filter_radius = 0.5;
struc.VolumeFraction = VolumeFraction;
% Normalization factors
struc.theta = 1;
struc.alpha = 1;
% problem parameters
struc.constant = 1;
struc.PowerKS = 8;
struc.SimpPenalty = 3;
struc.StressPower = 1;
struc.MinStressValue = 1e-3;
struc.StressNormFactor = ones(struc.numCells,1);

end