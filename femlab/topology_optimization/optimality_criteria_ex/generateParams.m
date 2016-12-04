function [usr_par] = generateParams(DomainSpecs, generateRHS)
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
%  INPUT:   nxint             structure
%                             relevant domain data
%
%           nPDEs             integer
%                             number of PDEs to be solve
%
%           generateRHS       functor
%                             right-hand side functor      
%
%  OUTPUT:  usr_par     contains all input parameters, as well as additional
%                       computed quantities
%
%  NOTE: Please read in-code comments for a precise explanation of the
%        computed data fields of the usr_par structure.
%
%  AUTHORS: Miguel Aguilo
%           (Sandia National Labs)

set(0, 'defaultaxesfontsize',12,'defaultaxeslinewidth',0.7,...
    'defaultlinelinewidth',0.8,'defaultpatchlinewidth',0.7,...
    'defaulttextfontsize',12)

%%%%%%%%%%% General input data
usr_par.DomainSpecs = DomainSpecs;
[usr_par] = generateProblemSpecs(usr_par);

%%%%%%%%%%% Get the number of vertices in the grid and the number of working cells
coarse_level = 0;
[usr_par] = generateMesh(usr_par, coarse_level);

%%%%%%%%%%% get cell and subcell cubature poitns and weights
[usr_par] = getCubature(usr_par);

%%%%%%%%%%% Generate problem specific data.
[usr_par] = generateProbSpecificData(usr_par);

%%%%%%%%%%% Generate the right hand sides and dirichlet boundary data
[usr_par] = generateRHS(usr_par);

usr_par.G = usr_par.G*ones(usr_par.nVertGrid,1);
usr_par.B = usr_par.B*ones(usr_par.nVertGrid,1);

end
