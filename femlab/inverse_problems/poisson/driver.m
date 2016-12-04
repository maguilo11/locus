% Driver for Inverse Poisson
clc;
%clear;

% Algorithm, testing, and problem interface directories
addpath /Users/miguelaguilo/dotk/matlab/exe/;
addpath /Users/miguelaguilo/dotk/matlab/mfiles/;
addpath /Users/miguelaguilo/Research/intrelab;
addpath /Users/miguelaguilo/Research/intrelab/mesh2/;
addpath /Users/miguelaguilo/Research/femlab/tools/;
%addpath /Users/miguelaguilo/Research/femlab/inverse/poisson/interfaceL2/
%addpath /Users/miguelaguilo/Research/femlab/inverse/poisson/interfaceResidual/
addpath /Users/miguelaguilo/Research/femlab/inverse/poisson/interfaceResidualNoSolve/

global GLB_INVP;

% Introduce the problem
fprintf('\n*** Inverse Poisson - Compliance Error Minimization ***\n');

% Gradient based optimization algorithm.
% Options: 1) NonLinearCG 
%          2) QuasiNewton
%          3) NewtonTypeLS 
%          4) NewtonTypeTR
%          5) IxNewtonTypeLS 
%          6) IxNewtonTypeTR 
%          7) IxSqpTypeTR
%          8) OptimalityCriteria
%          9) MMA
%          10) LinMoreNewtonTR
%          11) IxLinMoreNewtonTR
%          12) KelleySachsNewtonTR
%          13) IxKelleySachsNewtonTR
%          14) Diagnostics
Inputs.SolutionType = 'NonLinearCG';
% Problem Type
% Options: 1) ULP - UNCONSTRAINED LINEAR PROGRAMMING
%          2) UNLP - UNCONSTRAINED NONLINEAR PROGRAMMING 
%          3) ELP - EQUALITY CONSTRAINED LINEAR PROGRAMMING
%          4) ENLP - EQUALITY CONSTRAINED NONLINEAR PROGRAMMING
%          5) LP_BOUND - BOUND CONSTRAINED LINEAR PROGRAMMING
%          6) NLP_BOUND - BOUND CONSTRAINED NONLINEAR PROGRAMMING
%          7) ELP_BOUND - BOUND+EQUALITY CONSTRAINED LINEAR PROGRAMMING
%          8) ENLP_BOUND - BOUND+EQUALITY CONSTRAINED NONLINEAR PROGRAMMING
%          9) CLP - GENERAL CONSTRAINED LINEAR PROGRAMMING (EQUALITY+INEQUALITY)
%          10) CNLP - GENERAL CONSTRAINED NONLINEAR PROGRAMMING (EQUALITY+INEQUALITY)
%          11) ILP - INEQUALITY CONSTRAINED LINEAR PROGRAMMING
Inputs.ProblemType = 'LP_BOUND';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Domain.xmin = -1;  % min dim in x-dir
Domain.xmax = 1;   % max dim in x-dir
Domain.ymin = -1;  % min dim in y-dir
Domain.ymax = 1;   % max dim in y-dir
Domain.nx = 100;     % num intervals in x-dir
Domain.ny = 100;     % num intervals in y-dir
% choice of synthetic data
% 1 = egg; 2 = 1D/2D/3D sphere; 3 = 1D/2D/3D sphere in cube
target_control = 'sphere_in_cube';
% regularization parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV'; 
%beta = 1e-1;
%gamma = 1e-1;
beta = 1e-7;
gamma = 1e-7;
% the regularization parameter beta is related to the level of noise
% the larger the noise, the larger beta should be
rng(0,'v5normal');
noise = 0.05;

rhs_fn=@(struc)generateSine(struc);

% generate problem-specific constant quantities
[GLB_INVP] = generateParams(Domain, rhs_fn, target_control, noise);
GLB_INVP.reg = reg;
GLB_INVP.beta = beta;
GLB_INVP.gamma = gamma;

% Set Parameter Space Dimension
Inputs.NumberDuals = GLB_INVP.nVertGrid;
Inputs.NumberStates = GLB_INVP.nVertGrid;
Inputs.NumberControls = GLB_INVP.nVertGrid;

% Set index matrix (used in examples with sparse data)
GLB_INVP.IndexMat = speye(Inputs.NumberStates,Inputs.NumberStates);
%GLB_INVP.IndexMat(1:10:Inputs.NumberStates,1:100:Inputs.NumberStates) = 0;

% Set Control Initial Gueass & Bounds
Inputs.InitialDual = zeros(Inputs.NumberDuals,1);
Inputs.InitialState = zeros(Inputs.NumberStates,1);
Inputs.ControlLowerBounds = 1e-6*ones(Inputs.NumberControls,1);
Inputs.ControlUpperBounds = 1*ones(Inputs.NumberControls,1);
Inputs.InitialControl = ...
    (Inputs.ControlLowerBounds + Inputs.ControlUpperBounds) ./ 2;
rhs = rhs_fn(GLB_INVP);
[state,K0] = solvePDE(GLB_INVP,Inputs.InitialControl,rhs);
residual = K0*GLB_INVP.exp_state - GLB_INVP.force;
potential_energy = state'*residual - GLB_INVP.exp_state'*residual;
GLB_INVP.theta = 1 / (potential_energy'*potential_energy);
data_misfit = (state-GLB_INVP.exp_state);
GLB_INVP.alpha = 1 / (data_misfit'*(GLB_INVP.M*data_misfit));
clear state K0;

% Set DOTk Algorithm/Problem Options
[Options, Operators] = setOptions(Inputs);
Operators.EqualityConstraint = equalityConstraint();

fprintf(1,' \n');
fprintf(1, ' Solve Inverse Poisson Problem \n');
fprintf(1,' Problem parameters: \n');
fprintf(1,' Number of intervals in x-dir = %4d \n', Domain.nx);
fprintf(1,' Number of intervals in y-dir = %4d \n', Domain.ny);
fprintf(1,' Number of state variables    = %4d \n', Inputs.NumberStates);
fprintf(1,' Number of control variables  = %4d \n', Inputs.NumberControls);
fprintf(1,' Number of dual variables     = %4d \n', Inputs.NumberDuals);
fprintf(1,' Regularization weight        = %12.6e \n', beta);
if(strcmp(reg,'TV'))
    fprintf(1,' Modified TV parameter gamma  = %12.6e \n', gamma);
end
fprintf(1,' Random normal noise          = %12.6e \n\n', noise);

% Solve Optimization Problem
[Output,Primal,TimeData] = getMin(Options,Operators);

% Output the run times
fprintf('\nSummary:\n--------\n');
fprintf('\nCPU TIME ---------------------------------> %4.2f seconds \n', ...
    TimeData.proctime);
fprintf('WALL CLOCK TIME --------------------------> %4.2f seconds \n', ...
    TimeData.walltime);

%----------------------------------------------------------------------
% Plot controls, states
%----------------------------------------------------------------------
if (~strcmp(Inputs.SolutionType,'Diagnostics'))
    plotResults(Output,Primal);
end