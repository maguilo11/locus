function [struc] = generateParams(DomainSpecs, generateRHS)
%  function [u] = generateParams(nxint, nyint, noise, nPDEs, generateRHS, pdesolver)
%
%  PURPOSE: Generates problem-specific constant quantities, which require
%           some computational work, but can be computed once and reused,
%           thus speeding up the overall computation. These can include
%           various types of data, and depend entirely on the problem
%           (for example, discretization data).
%           Also, all domain decomposition constructs are computed.
%
%  where u is the solution of the PDE:
%
%     - div( dev_sigma + vol_sigma ) = f      in Omega
%                                  u = u_D    on Gamma_D
%         (dev_sigma + vol_sigma)'*n = g      on Gamma_N
%                          dev_sigma = 2mu*( grad(u) -1/3 tr(grad(u)I )
%                          vol_sigma = kappa*tr(grad(u))I
%
%  The problem domain Omega is the square (xmin,xmax)x(ymin,ymax).
%
%  u     - state
%  mu    - shear modulus
%  kappa - bulk modulus
%  f     - source term
%
%  INPUT:   DomainSpecs       Domain specifications
%
%           generateRHS       functor right-hand side functor      
%
%  OUTPUT:  struc      struct with all input parameters and additional
%                      information for analysis
%
%  NOTE: Please read in-code comments for a precise explanation of the
%        computed data fields of the usr_par structure.
%
%  AUTHORS: Miguel Aguilo
%           (Sandia National Labs)

set(0, 'defaultaxesfontsize',12,'defaultaxeslinewidth',0.7,...
    'defaultlinelinewidth',0.8,'defaultpatchlinewidth',0.7,...
    'defaulttextfontsize',12)

%%%%%%%%%%% Get the number of vertices in the grid and the number of working cells
struc.DomainSpecs = DomainSpecs;
[struc] = generateMesh(struc);

%%%%%%%%%%% get cell and subcell cubature poitns and weights
[struc] = getCubature(struc);

%%%%%%%%%%% Generate problem specific data.
[struc] = generateProblemSpecs(struc);
[struc] = generateProbSpecificData(struc);

%%%%%%%%%%% Generate the right hand sides and dirichlet boundary data
[struc] = generateRHS(struc);

end
