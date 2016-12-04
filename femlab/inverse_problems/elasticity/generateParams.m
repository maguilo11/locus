function [struc] = generateParams(DomainSpecs, generateRHS, target_control, noise)
%  function [struc] = generateParams(DomainSpecs, generateRHS, target_control, noise)
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
%  u_exp - observed state (experiments)
%  f     - source term
%
%  INPUT:   DomainSpecs       Domain specifications
%
%           generateRHS       functor right-hand side functor
%
%           target_control    char-type - target control data
%
%           noise             noise value for experimental data
%
%  OUTPUT:  struc      struct with all input parameters and additional
%                      information for analysis
%
%  NOTE: Please read in-code comments for a precise explanation of the
%        computed data fields of the usr_par structure.
%
%  AUTHORS: Miguel Aguilo
%           Dookie's Corp.
%           August 16, 2011

set(0, 'defaultaxesfontsize',12,'defaultaxeslinewidth',0.7,...
    'defaultlinelinewidth',0.8,'defaultpatchlinewidth',0.7,...
    'defaulttextfontsize',12)

%%%%%%%%%% General input data
spaceDim      = 2;                  % physical spatial dimensions for cell (element)
sideDim       = 1;                  % physical spatial dimensions for subcell (side)
cellType      = 'Triangle';         % cell (element) topology
nVert         = 3;                  % number of cell vertices
cubDegree     = 3;                  % max. degree of the polynomial that can be represented by the basis
numFields     = 3;                  % number of fields (i.e. number of basis functions)
sideType      = 'Line';             % subcell (side) topology
nVertSide     = 2;                  % number of subcell vertices
cubDegreeSide = 2;                  % subcells max. degree of the polynomial that can be represented by the basis
numFieldsSide = 2;                  % subcells number of fields (i.e. number of basis functions)
numSides      = 3;                  % number of sides for one cell
numDof        = spaceDim*numFields; % number of element degrees of freedom

struc.spaceDim      = spaceDim;
struc.sideDim       = sideDim;
struc.cellType      = cellType;
struc.sideType      = sideType;
struc.nVert         = nVert;
struc.nVertSide     = nVertSide;
struc.cubDegree     = cubDegree;
struc.cubDegreeSide = cubDegreeSide;
struc.numFieldsSide = numFieldsSide;
struc.numFields     = numFields;
struc.numSides      = numSides;
struc.nxint         = DomainSpecs.nx;
struc.nyint         = DomainSpecs.ny;
struc.numDof        = numDof;

%%%%%%%%%% Generate computational mesh on [xmin,xmax]x[ymin,ymax]
[mesh] = RectGrid( DomainSpecs.xmin, DomainSpecs.xmax, DomainSpecs.ymin, ...
    DomainSpecs.ymax, DomainSpecs.nx, DomainSpecs.ny);
nVertGrid = max( max(mesh.t(:,1:3)) );
numCells  = size(mesh.t, 1);

%%%%%%%%%% Generate mesh in Intrepid format
[~, iIdx, jIdx, ~] = generateIntrepidMesh(mesh, nVert, numCells);
struc.iIdxVertices = iIdx;
struc.jIdxVertices = jIdx;
struc.iIdxMix      = [(2*iIdx)-1; 2*iIdx];
struc.iIdxMix      = reshape(struc.iIdxMix,1,numel(struc.iIdxMix));
struc.jIdxMix      = [jIdx; jIdx];
struc.jIdxMix      = reshape(struc.jIdxMix,1,numel(struc.jIdxMix));

%%%%%%%%%% Generate mesh data sructures for solid mechanics problems
[cellNodes, iIdx, jIdx, iVecIdx] = ...
    generateIntrepidMeshElasticity(mesh, nVert, numCells);
struc.mesh       = mesh;
struc.numCells   = numCells;
struc.nVertGrid  = nVertGrid;
struc.cellNodes  = cellNodes;
struc.iIdxDof    = iIdx;
struc.jIdxDof    = jIdx;
struc.iVecIdxDof = iVecIdx;

%%%%%%%%%%% get cell and subcell cubature poitns and weights
[struc] = getCubature(struc);

%%%%%%%%%%% Generate problem specific data.
[struc] = generateProbSpecificData(struc);

%%%%%%%%%%% Generate the right hand sides and dirichlet boundary data
[struc] = generateRHS(struc);

%%%%%%%%%%% Generate finer mesh on [xmin,xmax]x[ymin,ymax] to generate experimental data
coarse_level = 1;
nxint_fine = DomainSpecs.nx*(2^coarse_level);
nyint_fine = DomainSpecs.ny*(2^coarse_level);
[mesh_fine] = RectGrid( DomainSpecs.xmin, DomainSpecs.xmax, DomainSpecs.ymin, ...
    DomainSpecs.ymax, nxint_fine, nyint_fine);
struc_fine.mesh      = mesh_fine;
struc_fine.numCells  = size(mesh_fine.t, 1);
struc_fine.nVertGrid = max( max(mesh_fine.t(:,1:3)) );

%%%%%%%%%%% Generate mesh in Intrepid format
[~, iIdx, jIdx, ~] = ...
    generateIntrepidMesh(mesh_fine, nVert, struc_fine.numCells);

struc_fine.iIdxVertices = iIdx;
struc_fine.jIdxVertices = jIdx;
struc_fine.iIdxMix      = [(2*iIdx)-1; 2*iIdx];
struc_fine.iIdxMix      = ...
    reshape(struc.iIdxMix,1,numel(struc.iIdxMix));
struc_fine.jIdxMix      = [jIdx; jIdx];
struc_fine.jIdxMix      = ...
    reshape(struc.jIdxMix,1,numel(struc.jIdxMix));

%%%%%%%%%% Generate mesh data sructures for solid mechanics problems
[cellNodes, iIdx, jIdx, iVecIdx] = ...
    generateIntrepidMeshElasticity(mesh_fine, nVert, struc_fine.numCells);

struc_fine.cellNodes  = cellNodes;
struc_fine.iIdxDof    = iIdx;
struc_fine.jIdxDof    = jIdx;
struc_fine.iVecIdxDof = iVecIdx;

struc_fine.spaceDim      = spaceDim;
struc_fine.sideDim       = sideDim;
struc_fine.cellType      = cellType;
struc_fine.sideType      = sideType;
struc_fine.nVert         = nVert;
struc_fine.nVerSide      = nVertSide;
struc_fine.cubDegree     = cubDegree;
struc_fine.cubDegreeSide = cubDegreeSide;
struc_fine.numFieldsSide = numFieldsSide;
struc_fine.numFields     = numFields;
struc_fine.numSides      = numSides;
struc_fine.nxint         = nxint_fine;
struc_fine.nyint         = nyint_fine;
struc_fine.numDof        = numDof;
struc_fine.pdata         = target_control;

%%%%%%%%%%% get cell and subcell cubature poitns and weights
[struc_fine] = getCubature(struc_fine);

%%%%%%%%%%% Generate problem specific data.
[struc_fine] = generateProbSpecificData(struc_fine);

%%%%%%%%%%% Generate the right hand sides and dirichlet boundary data
[struc_fine] = generateRHS(struc_fine);

%%%%%%%%%%% Generate the true material parameters
[shear_fine, bulk_fine] = generateTrueNodalElasticParameters(struc_fine);

clear u_exp k_exp;
%%%%%%%%%%% Solve PDE
sigma = 0.05;
[state_fine_org,~] = solvePDE(struc_fine, shear_fine, bulk_fine);
random_nums = (sigma+(-sigma-sigma))*rand(size(state_fine_org));
state_fine = state_fine_org.*(1 + (noise.*random_nums));

%%%%%%%%%%% get experimental real state u_x
state_x_fine = state_fine(1:struc.spaceDim:end);
state_x_exp = getExperiments(state_x_fine, nxint_fine, ...
    nyint_fine, coarse_level, struc);

%%%%%%%%%%% get experimental real state u_y
state_y_fine = state_fine(2:struc.spaceDim:end);
state_y_exp = getExperiments(state_y_fine, nxint_fine, ...
    nyint_fine, coarse_level, struc);

%%%%%%%%%%% assemble state vector field
exp_state = zeros(1, struc.spaceDim*struc.nVertGrid);
exp_state(1:struc.spaceDim:end) = state_x_exp(1,:);
exp_state(2:struc.spaceDim:end) = state_y_exp(1,:);

struc.Mf         = struc_fine.Ms;
struc.exp_state  = exp_state';
struc.mu_fine    = shear_fine;
struc.kappa_fine = bulk_fine;
struc.mesh_fine  = mesh_fine;

clear usr_par_fine;

end
