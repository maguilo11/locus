% Driver for topology optimization problem in elasticity
clc;
%clear all;

% SQP algorithm, testing, and problem interface directories
addpath /Users/miguelaguilo/dotk/matlab/;
addpath /Users/miguelaguilo/dotk/matlab/exe/;
addpath /Users/miguelaguilo/dotk/matlab/mfiles/;
addpath /Users/miguelaguilo/Research/intrelab;
addpath /Users/miguelaguilo/Research/intrelab/mesh2/;
addpath /Users/miguelaguilo/Research/femlab/tools/;
addpath /Users/miguelaguilo/Research/femlab/optimality_criteria/interface/;

global GLB_INVP;

% Introduce the problem
fprintf('\n*** Elastosatics Topology Optimization ***\n');

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
%          10) SteihaugToint
%          11) IxSteihaugToint
%          12) Diagnostics
Inputs.SolutionType = 'Diagnostics';
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
Inputs.ProblemType = 'CNLP';
% Use Gauss-Newton?
GaussNewtonHessian = false;
% want plots?
iplot = true;
% Limit on volume
VolumeFraction = 0.5;
% Density model
model_t = 'simp';

% Domain specifications
Domain.xmin = 0;  % min dim in x-dir
Domain.xmax = 3;   % max dim in x-dir
Domain.ymin = 0;  % min dim in y-dir
Domain.ymax = 1;   % max dim in y-dir
Domain.nx = 60;     % num intervals in x-dir
Domain.ny = 20;     % num intervals in y-dir
% regularization parameters ( Tikhonov or TV (Total Variation) )
reg = 'TV'; 
beta = 1/(2*Domain.ny);
gamma = 1e-6;
% the regularization parameter beta is related to the level of noise
% the larger the noise, the larger beta should be
rng(0,'v5normal');
noise = 0.00;

rhs_fn=@(usr_par)generateNodalForce(usr_par);

% generate problem-specific constant quantities
[GLB_INVP] = generateParams(Domain, rhs_fn);
GLB_INVP.radius = 1;

% generate filtering operator H
%[GLB_INVP] = generateFilterOperator(GLB_INVP);

% store problem-specific constant quantites
GLB_INVP.reg = reg;
GLB_INVP.beta = beta;
GLB_INVP.gamma = gamma;
GLB_INVP.VolumeFraction = VolumeFraction;
GLB_INVP.model_t = model_t;
% Normalization factors
GLB_INVP.theta = 1;
GLB_INVP.alpha = 1;
% Gauss-Newton Hessian approximation?
GLB_INVP.gn = GaussNewtonHessian;

% Set Parameter Space Dimension
Inputs.NumberDuals = GLB_INVP.spaceDim * GLB_INVP.nVertGrid;
Inputs.NumberStates = GLB_INVP.spaceDim * GLB_INVP.nVertGrid;
Inputs.NumberControls = GLB_INVP.nVertGrid;

% Set Control Initial Gueass & Bounds
Inputs.InitialDual = zeros(Inputs.NumberDuals,1);
Inputs.InitialState = zeros(Inputs.NumberStates,1);
Inputs.InitialControl = ...
    GLB_INVP.VolumeFraction * ones(Inputs.NumberControls,1);
Inputs.ControlLowerBounds = 1e-3 * ones(Inputs.NumberControls,1);
Inputs.ControlUpperBounds = ones(Inputs.NumberControls,1);

% Set Main Optimization Algorithm/Problem Options
[Options, Operators] = setOptions(Inputs);

% Store Topology Optimization Problem Specific Parameters
GLB_INVP.SimpPenalty = 3;
one = ones(GLB_INVP.nVertGrid,1);
GLB_INVP.OriginalVolume = sum(GLB_INVP.Ms*one);
GLB_INVP.alpha = 1 / (sum(GLB_INVP.Ms*Inputs.InitialControl)^2);
GLB_INVP.CellStifsnessMat = computeStiffnessMatrix(GLB_INVP.G, GLB_INVP.B);
[state] = Operators.EqualityConstraint.solve(one);
StiffMat = reshape(GLB_INVP.CellStifsnessMat, 1, numel(GLB_INVP.CellStifsnessMat));
K = sparse(GLB_INVP.iIdxDof, GLB_INVP.jIdxDof, StiffMat);
GLB_INVP.theta = 0.5 / (state'*K*state);
clear K;

fprintf(1,' \n');
fprintf(1, ' Solve Linear Elastostatic topology optimization probelm \n');
fprintf(1,' Problem parameters: \n');
fprintf(1,' Number of intervals in x-dir = %4d \n', Domain.nx);
fprintf(1,' Number of ntervals in y-dir  = %4d \n', Domain.ny);
fprintf(1,' Number of state variables    = %4d \n', Inputs.NumberStates);
fprintf(1,' Number of control variables  = %4d \n', Inputs.NumberControls);
fprintf(1,' Number of dual variables     = %4d \n', Inputs.NumberDuals);
fprintf(1,' Gauss-Newton Hessian?        = %4d \n', GaussNewtonHessian);
fprintf(1,' Regularization weight        = %12.6e \n', beta);
if(strcmp(reg,'TV'))
    fprintf(1,' TV regularization parameter gamma = %12.6e \n', gamma);
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
% Plot parameters, states
%----------------------------------------------------------------------
if (iplot == true && ~strcmp(Inputs.SolutionType,'Diagnostics'))
    plotResults(Output,Primal);
end
