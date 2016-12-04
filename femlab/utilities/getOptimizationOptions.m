%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               MAIN OPTIMIZATION PROBLEM OPTIONS INPUTS                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options, Operators] = ...
    getOptimizationOptions(algorithm_t, problem_t, Inputs)
Options = [];
% Diagnostics Display Option
[Options] = setDiagnosticsDisplayOption(Options);
% Design Variables Information
Options.NumControls = Inputs.NumControls;
Options.NumStates = Inputs.NumStates;
Options.NumEqConstraints = Inputs.NumEqConstraints;
Options.NumIeqConstraints = Inputs.NumIeqConstraints;
% Bound Constraints
Options.BoundConstraints = 0;
[Options] = getLowerAndUpperBounds(Options);
% General Optimization Parameters
Options.MaxOptimizationItr = 150;
Options.ObjectiveFuncTol = 1e-12;
Options.GradientTol = 1e-10;
Options.TrialStepTol = 1e-12;
% Maximum Limited Memory Storage
Options.MaxLimitedMemoryStorage = 0;
switch algorithm_t
    case 'NonLinearCG'
        Options = getNonLinearCGOptions(Options);
    case 'QuasiNewton'
        Options = getQuasiNewtonOptions(Options);
    case 'LineSearchNewtonCG'
        Options = getLineSearchCGOptions(Options);
    case 'TrustRegionNewtonCG'
        Options = getTrustRegionNewtonCGOptions(Options);
    case 'InexactSQP'
        [Options] = getInexactSQPOptions(Options, Inputs);
    otherwise
        fprintf(' Invalid SOL Algorithm. See Users Manual. ');
end
% Set control variables initial guess
Options.Control = ones(Inputs.NumControls,1)*2;
% Set linear algebra
Options.LinearAlgebra = getLinearAlgebra;
% Get problem operators 
[Operators] = getOperators(problem_t);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         SET SOL DISPLAY OPTIONS                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setDiagnosticsDisplayOption(Options)
%   1. ITERATION
%   2. FINAL
%   3. OFF
Options.OutputDataDisplayOption = 'FINAL';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   SET REDUCED SPACE ALGORITHMS INPUTS                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = getNonLinearCGOptions(Options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options.SOL_Algorithm = 1; % DO NOT MODIFY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options = getSearchDirectionOptions(Options);
Options = getGradientOperatorOptions(Options);
Options = getLineSearchOptions(Options);
end

function [Options] = getQuasiNewtonOptions(Options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options.SOL_Algorithm = 2; % DO NOT MODIFY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options = getGradientOperatorOptions(Options);
Options = getInvHessianOperatorOptions(Options);
Options = getLineSearchOptions(Options);
end

function [Options] = getLineSearchCGOptions(Options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options.SOL_Algorithm = 3; % DO NOT MODIFY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options.MaxNumCGItr=200;
Options = getGradientOperatorOptions(Options);
Options = getHessianOperatorOptions(Options);
Options = getInvHessianOperatorOptions(Options);
Options = getLineSearchOptions(Options);
Options = getReducedSpaceLeftPrecOptions(Options);
Options = getReducedSpaceRightPrecOptions(Options);
end

function [Options] = getTrustRegionNewtonCGOptions(Options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options.SOL_Algorithm = 4; % DO NOT MODIFY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options.MaxNumCGItr=200;
Options = getGradientOperatorOptions(Options);
Options = getHessianOperatorOptions(Options);
Options = getInvHessianOperatorOptions(Options);
Options = getLineSearchOptions(Options);
Options = getTrustRegionOptions(Options);
Options = getReducedSpaceLeftPrecOptions(Options);
Options = getReducedSpaceRightPrecOptions(Options);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     SET FULL SPACE ALGORITHM INPUTS                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = getInexactSQPOptions(Options, Inputs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options.SOL_Algorithm = 5; % DO NOT MODIFY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options.TangentialSubProblemMaxItr = 200;
Options.MaxKrylovSolverItr = 5;
Options.KrylovRestartItr = Options.MaxKrylovSolverItr;
Options.ConstraintTol = 1e-10;
Options.QuasiNormalTol = 1e-4;
Options.TangentialTol = 1e-4;
Options.TangentialTolReductionFactor = 1e-3;
Options.TolReductionFactor = 1e-1;
Options.ProjectionTol = 1e-4;
Options.LagrangeMultipliersTol = 1e-4;
Options.LagrangeMultipliersGradTol = 1e4;
Options.OrthogonalityTol = 0.5;
Options.ActualOverPredictedReductionParam = 1e-8;
Options.QuasiNormalTrustRegionFractionParam = 0.8;
Options.MaxEffectiveTangentialOverTrialStepParam = 2;
% Other Options
Options = getTrustRegionOptions(Options);
Options = getHessianOperatorOptions(Options);
Options = getGradientOperatorOptions(Options);
Options = getFullSpaceRightPrecOptions(Options);
Options = getFullSpaceLeftPrecOptions(Options);
% Set state variables initial guess and corresponding output variable 
Options.State = ones(Inputs.NumStates,1);
% Set lagrange multipliers initial guess and corresponding output variable 
Options.LagrangeMultipliers = zeros(Inputs.NumEqConstraints,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             METHODS INPUTS                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = getTrustRegionOptions(Options)
% Set trust region options
Options.MaxTrustRegionItr = 5;
Options.TrustRegionRadius = 1e4;
Options.MaxTrustRegionRadius = 1e4;
Options.MinTrustRegionRadius = 1e-6;
Options.TrustRegionRadiusContractionParam = 0.5;
Options.TrustRegionRadiusExpansionParam = 2;
Options.AllowableMinimumActualReductionRatio = 0.01;
end

function [Options] = getLineSearchOptions(Options)
% Line Search Methods:
%   1. BACKTRACKING_ARMIJO
%   2. BACKTRACKING_GOLDSTEIN
%   3. BACKTRACKING_CUBIT_INTRP
%   4. GOLDENSECTION
Options.LineSearchMethod = 'BACKTRACKING_CUBIT_INTRP';
% General Line Search Options
Options.MaxLineSearchItr = 5;
%Options.LineSeachStepTol = 1e-3;
Options.LineSeachStepTol = 1e-4;
Options.LineSearchContractionFactor = 0.1;
end

function [Options] = getGradientOperatorOptions(Options)
% Gradient Operator Options: 
%   1. FORWARD_DIFFERENCE_GRAD
%   2. BACKWARD_DIFFERENCE_GRAD
%   3. CENTRAL_DIFFERENCE_GRAD
%   4. USER_DEFINED_GRAD
%   5. PARALLEL_FORWARD_DIFFERENCE_GRAD
%   6. PARALLEL_BACKWARD_DIFFERENCE_GRAD
%   7. PARALLEL_CENTRAL_DIFFERENCE_GRAD
Options.GradientOperator = 'USER_DEFINED_GRAD';
if (~strcmp(Options.GradientOperator,'USER_DEFINED_GRAD'))
    Options.FiniteDifferencePerturbation = [1e-5, 1e-6];
end

end

function [Options] = getInvHessianOperatorOptions(Options)
% Inverse Hessian Operator Options: 
%   1. LBFGS_INV_HESS
%   2. LDFP_INV_HESS
%   3. LSR1_INV_HESS
%   4. SR1_INV_HESS
%   5. BFGS_INV_HESS
%   6. IDENTITY_INV_HESS
%   7. USER_DEFINED_INV_HESS
Options.InvHessianOperator = 'IDENTITY_INV_HESS';
end

function [Options] = getHessianOperatorOptions(Options)
% Hessian Operator Options: 
%   1. LBFGS_HESS
%   2. LDFP_HESS
%   3. LSR1_HESS
%   4. SR1_HESS
%   5. DFP_HESS
%   6. IDENTITY_HESS
%   7. USER_DEFINED_HESS
Options.HessianOperator = 'USER_DEFINED_HESS';
end

function [Options] = getSearchDirectionOptions(Options)
% Search Direction Method
%   1. FLETCHER REEVES
%   2. POLAK RIBIERE
%   3. HESTENES STIEFEL
Options.SearchDirectionMethod = 'HESTENES STIEFEL';
end

function [Options] = getLowerAndUpperBounds(Options)
% Bound Constraints 
%   0=None
%   1=Controls
%   2=States
%   3=Controls & States
switch Options.BoundConstraints
    case 0
    case 1
        Options.LowerBounds = [0 0];
        Options.UpperBounds = [5 5];
    case 2
        % State
    case 3
        % State & Control
end
end

function [Options] = getFullSpaceRightPrecOptions(Options)
% Preconditioner Types:
%   1. NO_PREC
%   2. USER_DEFINED
Options.RightPreconditionerType = 'NO_PREC';
end

function [Options] = getReducedSpaceRightPrecOptions(Options)
% Preconditioner Types:
%   1. NO_PREC
%   2. USER_DEFINED_PREC
Options.RightPreconditionerType = 'NO_PREC';
end

function [Options] = getFullSpaceLeftPrecOptions(Options)
% Preconditioner Types:
%   1. NO_PREC
%   2. USER_DEFINED_PREC
%   3. FULL_SCHUR_PREC
%   4. INCOMPLETE_SCHUR_PREC
Options.LeftPreconditionerType = 'FULL_SCHUR_PREC';
end

function [Options] = getReducedSpaceLeftPrecOptions(Options)
% Preconditioner Types:
%   1. NO_PREC
%   2. USER_DEFINED_PREC
Options.LeftPreconditionerType = 'NO_PREC';
end
