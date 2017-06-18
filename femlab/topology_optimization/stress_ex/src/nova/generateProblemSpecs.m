function [struc] = generateProblemSpecs(struc)

% Material Properties
E = 8;
nu = 0.3;
stiffness_bound = 1e-6;
%
struc.B = E / (3*(1 - (2*nu)));  % bulk modulus
struc.G = E / (2*(1+nu));          % shear modulus
struc.G = struc.G*ones(struc.nVertGrid,1);
struc.B = struc.B*ones(struc.nVertGrid,1);
struc.Bmin = stiffness_bound * struc.B;
struc.Gmin = stiffness_bound * struc.G;

end