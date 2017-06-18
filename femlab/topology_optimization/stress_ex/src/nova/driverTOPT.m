function [struc] = driverTOPT(mesh_file)

% Limit on volume
VolumeFraction = 0.4;

% perimeter control parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV'; 
beta = 1e-6; 
gamma = 1e-6;

% generate problem-specific constant quantities
input.neumann = false;
input.mesh_file = mesh_file;
rhs_fn=@(struc)generateNodalForce(struc);
[struc] = generateParams(input, rhs_fn);

% store problem-specific constant quantites
struc.reg = reg;
struc.beta = beta;
struc.gamma = gamma;
struc.filter_radius = 4;
struc.VolumeFraction = VolumeFraction;
% Normalization factors
struc.theta = 1;
struc.alpha = 1;
% problem parameters
struc.constant = 1;
struc.PowerKS = 16;
struc.SimpPenalty = 6;
struc.StressPower = 1;
struc.MinStressValue = 1e-6;
struc.StressNormFactor = ones(struc.numCells,1);

end