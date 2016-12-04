function [struc] = generateParams(DomainSpecs, generateRHS, target_control, noise)

%  function [u] = generateParams(DomainSpecs, generateRHS, target_control)
%
%  PURPOSE: Generates problem-specific constant quantities, which require
%           some computational work, but can be computed once and reused,
%           thus speeding up the overall computation. These can include
%           various types of data, and depend entirely on the problem
%           (for example, discretization data).
%           Also, all domain decomposition constructs are computed.
%
%  where y is the solution of the PDE:
%
%  - div(k grad(u))  = f              in Omega
%                  u = u_D            on Gamma_D
%     (k grad(u))'*n = g              on Gamma_N
%
%  The problem domain Omega is the square (xmin,xmax)x(ymin,ymax).
%
%  u         - state
%  k         - control
%  f         - source term
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
%           March 30, 2011

set(0, 'defaultaxesfontsize',12,'defaultaxeslinewidth',0.7,...
    'defaultlinelinewidth',0.8,'defaultpatchlinewidth',0.7,...
    'defaulttextfontsize',12)

%%%% General input data
spaceDim      = 2;           % physical spatial dimensions
cellType      = 'Triangle';  % cell type
nVert         = 3;           % number of cell vertices
cubDegree     = 3;           % max. degree of the polynomial that can be represented by the basis
numFields     = 3;           % number of fields (i.e. number of basis functions)
sideType      = 'Line';      % subcell (side) topology
sideDim       = 1;           % physical spatial dimensions for subcell (side)
nVertSide     = 2;           % number of subcell vertices
cubDegreeSide = 2;           % subcells max. degree of the polynomial that can be represented by the basis
numFieldsSide = 2;           % subcells number of fields (i.e. number of basis functions)
numSides      = 3;                  % number of sides for one cell

struc.spaceDim      = spaceDim;
struc.cellType      = cellType;
struc.nVert         = nVert;
struc.cubDegree     = cubDegree;
struc.numFields     = numFields;
struc.sideDim       = sideDim;
struc.sideType      = sideType;
struc.nVertSide     = nVertSide;
struc.cubDegreeSide = cubDegreeSide;
struc.numFieldsSide = numFieldsSide;
struc.numSides      = numSides;
struc.nxint         = DomainSpecs.nx;
struc.nyint         = DomainSpecs.ny;

%%%% Generate computational mesh on [xmin,xmax]x[ymin,ymax]
[mesh] = RectGrid( DomainSpecs.xmin, DomainSpecs.xmax, DomainSpecs.ymin, ...
                   DomainSpecs.ymax, DomainSpecs.nx, DomainSpecs.ny);
nVertGrid = max( max(mesh.t(:,1:3)) );
numCells  = size(mesh.t, 1);

%%%% Generate mesh in Intrepid format
[cellNodes, iIdx, jIdx, iVecIdx] = ...
    generateIntrepidMesh(mesh, nVert, numCells);

struc.mesh      = mesh;
struc.numCells  = numCells;
struc.nVertGrid = nVertGrid;
struc.cellNodes = cellNodes;
struc.iIdx      = iIdx;
struc.jIdx      = jIdx;
struc.iVecIdx   = iVecIdx;

%%%%%%%%%%% get cell and subcell cubature poitns and weights
[struc] = getCubature(struc);

%%%% Generate problem specific data.
[struc] = generateProbSpecificData(struc);

%%%% Generate the right hand sides.
force = generateRHS(struc);
struc.force = force;

%%%% Generate finer mesh on [xmin,xmax]x[ymin,ymax] to generate experimental data
coarse_level = 1;
nxint_fine = DomainSpecs.nx*(2^coarse_level);
nyint_fine = DomainSpecs.ny*(2^coarse_level);
[mesh] = RectGrid( DomainSpecs.xmin, DomainSpecs.xmax, DomainSpecs.ymin, ...
                   DomainSpecs.ymax, nxint_fine, nyint_fine);
nVertGrid = max( max(mesh.t(:,1:3)) );
numCells  = size(mesh.t, 1);

%%%% Generate mesh in Intrepid format
[cellNodes, iIdx, jIdx, iVecIdx] = ...
    generateIntrepidMesh(mesh, nVert, numCells);

struc_fine.mesh          = mesh;
struc_fine.numCells      = numCells;
struc_fine.nVertGrid     = nVertGrid;
struc_fine.cellNodes     = cellNodes;
struc_fine.iIdx          = iIdx;
struc_fine.jIdx          = jIdx;
struc_fine.iVecIdx       = iVecIdx;
struc_fine.spaceDim      = spaceDim;
struc_fine.cellType      = cellType;
struc_fine.sideDim       = sideDim;
struc_fine.sideType      = sideType;
struc_fine.nVertSide     = nVertSide;
struc_fine.cubDegreeSide = cubDegreeSide;
struc_fine.numFieldsSide = numFieldsSide;
struc_fine.numSides      = numSides;
struc_fine.nVert         = nVert;
struc_fine.cubDegree     = cubDegree;
struc_fine.numFields     = numFields;
struc_fine.pdata         = target_control;
struc_fine.nxint          = DomainSpecs.nx;
struc_fine.nyint          = DomainSpecs.ny;

%%%%%%%%%%% get cell and subcell cubature poitns and weights
[struc_fine] = getCubature(struc_fine);

%%%% Generate problem specific data.
[struc_fine] = generateProbSpecificData(struc_fine);

%%%% Generate the right hand sides.
force_fine = generateRHS(struc_fine);

%%%% Generate the true solution
struc.pdata = target_control;
control_exp = generateTrueNodalParameters(struc);
control_fine_nodal = generateTrueNodalParameters(struc_fine);

clear exp_state;
%%%% Solve PDE
[u_fine_org,~] = solvePDE(struc_fine,control_fine_nodal,force_fine);
random_nums = (noise+(-noise-noise))*rand(size(u_fine_org));
u_fine = u_fine_org.*(1 + (random_nums));

%%%% Get experimental data
exp_state = ...
    getExperiments(u_fine, nxint_fine, nyint_fine, coarse_level, struc);
    
%%%% Store experimental state and control data
struc.Mf = struc_fine.M;
struc.exp_state   = exp_state;
struc.kappa_exp  = control_exp;
struc.mesh_fine   = struc_fine.mesh;
struc.kappa_fine  = control_fine_nodal;

clear usr_par_fine;

end
