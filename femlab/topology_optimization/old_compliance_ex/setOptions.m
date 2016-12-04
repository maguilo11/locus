%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               MAIN OPTIMIZATION PROBLEM OPTIONS INPUTS                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options, Operators] = setOptions(Inputs)
Options = [];
% Set Container Type
Options = setContainerType(Options);
% Set Algorithm Options
Options.SolutionType = Inputs.SolutionType;
switch Inputs.SolutionType
    case 'NonLinearCG'
        Options = setNonLinearCG(Inputs, Options);
    case 'QuasiNewton'
        Options = setQuasiNewton(Inputs, Options);
    case 'NewtonTypeLS'
        Options = setNewtonTypeLS(Inputs, Options);
    case 'NewtonTypeTR'
        Options = setNewtonTypeTR(Inputs, Options);
    case 'IxNewtonTypeLS'
        Options = setIxNewtonTypeLS(Inputs, Options);
    case 'IxNewtonTypeTR'
        Options = setIxNewtonTypeTR(Inputs, Options);
    case 'LinMoreNewtonTR'
        Options = setLinMoreNewtonTR(Inputs, Options);
    case 'IxLinMoreNewtonTR'
        Options = setIxLinMoreNewtonTR(Inputs, Options);
    case 'KelleySachsNewtonTR'
        Options = setKelleySachsNewtonTR(Inputs, Options);
    case 'IxKelleySachsNewtonTR'
        Options = setIxKelleySachsNewtonTR(Inputs, Options);
    case 'OptimalityCriteria'
        Options = setOptimalityCriteria(Inputs, Options);
    case 'MMA'
        Options = setMMA(Inputs, Options);
    case 'IxSqpTypeTR'
        Options = setInexactSQP(Inputs, Options);
    case 'Diagnostics'
        Options = setDiagnosticOptions(Inputs, Options);
    otherwise
        error(' Invalid Algorithm Type. See Users Manual. ');
end
% Diagnostics Display Option
Options = setDiagnosticsDisplayOption(Options);
% Set Problem Operators 
[Operators] = setOperators(Inputs);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            ALGORITHMS OPTIONS                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setNonLinearCG(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setGeneralOptions(Options);
Options = setNonLinearCGMethod(Options);
Options = setGradientComputationMethod(Options);
Options = setLineSearchMethod(Options);
end

function [Options] = setQuasiNewton(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setGeneralOptions(Options);
Options = setGradientComputationMethod(Options);
Options = setQuasiNewtonMethod(Options);
Options = setLineSearchMethod(Options);
end

function [Options] = setNewtonTypeLS(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setGeneralOptions(Options);
Options = setKrylovSolver(Options);
Options = setGradientComputationMethod(Options);
Options = setHessianComputationMethod(Options);
Options = setLineSearchMethod(Options);
end

function [Options] = setNewtonTypeTR(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setGeneralOptions(Options);
Options = setKrylovSolver(Options);
Options = setTrustRegionMethod(Options);
Options = setGradientComputationMethod(Options);
Options = setHessianComputationMethod(Options);
end

function [Options] = setIxNewtonTypeLS(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setGeneralOptions(Options);
Options = setKrylovSolver(Options);
Options = setLineSearchMethod(Options);
Options = setGradientComputationMethod(Options);
Options = setHessianComputationMethod(Options);
Options = setNumericalDifferentiationMethod(Options);
end

function [Options] = setIxNewtonTypeTR(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setGeneralOptions(Options);
Options = setKrylovSolver(Options);
Options = setTrustRegionMethod(Options);
Options = setGradientComputationMethod(Options);
Options = setHessianComputationMethod(Options);
Options = setNumericalDifferentiationMethod(Options);
end

function [Options] = setLinMoreNewtonTR(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setKrylovSolver(Options);
Options = setGeneralOptions(Options);
Options = setTrustRegionMethod(Options);
Options = setGradientComputationMethod(Options);
Options = setHessianComputationMethod(Options);
end

function [Options] = setIxLinMoreNewtonTR(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setKrylovSolver(Options);
Options = setGeneralOptions(Options);
Options = setTrustRegionMethod(Options);
Options = setGradientComputationMethod(Options);
Options = setNumericalDifferentiationMethod(Options);
end

function [Options] = setKelleySachsNewtonTR(Inputs, Options)
Options.MaxNumberUpdates = 10;
Options = setProblem(Inputs, Options);
Options = setKrylovSolver(Options);
Options = setGeneralOptions(Options);
Options = setTrustRegionMethod(Options);
Options = setGradientComputationMethod(Options);
Options = setHessianComputationMethod(Options);
end

function [Options] = setIxKelleySachsNewtonTR(Inputs, Options)
Options.MaxNumberUpdates = 10;
Options = setProblem(Inputs, Options);
Options = setKrylovSolver(Options);
Options = setGeneralOptions(Options);
Options = setTrustRegionMethod(Options);
Options = setGradientComputationMethod(Options);
Options = setNumericalDifferentiationMethod(Options);
end

function [Options] = setOptimalityCriteria(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setGeneralOptions(Options);
Options = setGradientComputationMethod(Options);
% Set Optimality Criteria (OC) Options
Options.MoveLimit = 2e-1;
Options.DualLowerBound = 0;
Options.DualUpperBound = 1e5;
Options.DampingParameter = 0.25;
Options.BisectionTolerance = 1e-3;
Options.MaxControlRelativeDiffTolerance = 1e-2;
end

function [Options] = setMMA(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setGeneralOptions(Options);
Options = setAsymptoteUpdateRule(Options);
Options = setGradientComputationMethod(Options);
% Set MMA Dual Solver Options
Options.MaxNumDualProblemItr = 25;
Options.DualSolver = 'HESTENES_STIEFEL';
Options.DualProbLineSearchMethod = 'CUBIC_INTRP';
Options.MaxNumLineSearchItr = 25;
Options.LineSearchStagnationTolerance = 1e-6;
Options.LineSearchContractionFactor = 0.5;
Options.DualProbBoundConstraintMethod = 'FEASIBLE_DIR';
Options.MaxNumFeasibleItr = 50;
Options.BoundConstraintStepSize = 0.5;
Options.BoundConstraintContractionFactor = 0.5;
% Dual Variable Options
Options.Dual = 1e-2 * ones(1,Inputs.NumberDuals);
Options.DualLowerBounds = 1e-10 * ones(1,Inputs.NumberDuals);
Options.DualUpperBounds = 1e0 * ones(1,Inputs.NumberDuals);
% Set MMA Options
Options.FeasibilityTolerance = 1e-4;
Options.OptimalityProximityTolerance = 1e-2;
Options.ExpectedOptimalObjectiveFunctionValue = 0.;
end

function [Options] = setInexactSQP(Inputs, Options)
Options = setProblem(Inputs, Options);
Options = setGeneralOptions(Options);
Options = setTrustRegionMethod(Options);
Options = setHessianComputationMethod(Options);
Options = setSqpLeftPreconditioner(Options);
Options = setGradientComputationMethod(Options);
% Set Maximum Number of Krylov Solvers Iterations
Options.MaxNumDualProblemItr = 200;
Options.MaxNumTangentialProblemItr = 200;
Options.MaxNumQuasiNormalProblemItr = 200;
Options.MaxNumTangentialSubProblemItr = 200;
% Set SQP Options
Options.TangentialTolerance = 1e-4;
Options.DualProblemTolerance = 1e-4;
Options.DualDotGradientTolerance = 1e4;
Options.MeritFunctionPenaltyParameter = 1;
Options.ToleranceContractionFactor = 1e-1;
Options.PredictedReductionParameter = 1e-8;
Options.QuasiNormalProblemRelativeTolerance = 1e-4;
Options.TangentialToleranceContractionFactor = 1e-3;
Options.ActualOverPredictedReductionTolerance = 1e-8;
Options.MaxEffectiveTangentialOverTrialStepRatio = 2;
Options.TangentialSubProbLeftPrecProjectionTolerance = 1e-4;
Options.QuasiNormalProblemTrustRegionRadiusPenaltyParameter = 0.8;
end

function [Options] = setDiagnosticOptions(Inputs, Options)
% 1) ULP - UNCONSTRAINED LINEAR PROGRAMMING
% 2) UNLP - UNCONSTRAINED NONLINEAR PROGRAMMING 
% 3) ELP - EQUALITY CONSTRAINED LINEAR PROGRAMMING
% 4) ENLP - EQUALITY CONSTRAINED NONLINEAR PROGRAMMING
% 5) LP_BOUND - BOUND CONSTRAINED LINEAR PROGRAMMING
% 6) NLP_BOUND - BOUND CONSTRAINED NONLINEAR PROGRAMMING
% 7) ELP_BOUND - BOUND+EQUALITY CONSTRAINED LINEAR PROGRAMMING
% 8) ENLP_BOUND - BOUND+EQUALITY CONSTRAINED NONLINEAR PROGRAMMING
% 9) CLP - GENERAL CONSTRAINED LINEAR PROGRAMMING (EQUALITY+INEQUALITY)
% 10) CNLP - GENERAL CONSTRAINED NONLINEAR PROGRAMMING (EQUALITY+INEQUALITY)
% 11) ILP - INEQUALITY CONSTRAINED LINEAR PROGRAMMING
%
Options.checkSecondDerivative = true;
Options.ProblemType = Inputs.ProblemType;
Options.FiniteDifferenceDiagnosticsLowerSuperScripts = -3;
Options.FiniteDifferenceDiagnosticsUpperSuperScripts = 4;

switch Options.ProblemType
    case 'ULP'
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    case 'UNLP'
        Options.NumberDuals = Inputs.NumberDuals;
        Options.Dual = rand(1,Inputs.NumberDuals);
        Options.NumberStates = Inputs.NumberStates;
        Options.State = rand(1,Inputs.NumberStates);
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    case 'ELP'
        Options.NumberDuals = Inputs.NumberDuals;
        Options.Dual = rand(1,Inputs.NumberDuals);
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    case 'ENLP'
        Options.NumberDuals = Inputs.NumberDuals;
        Options.Dual = rand(1,Inputs.NumberDuals);
        Options.NumberStates = Inputs.NumberStates;
        Options.State = rand(1,Inputs.NumberStates);
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    case 'LP_BOUND'
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    case 'NLP_BOUND'
        Options.NumberDuals = Inputs.NumberDuals;
        Options.Dual = rand(1,Inputs.NumberDuals);
        Options.NumberStates = Inputs.NumberStates;
        Options.State = rand(1,Inputs.NumberStates);
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    case 'ELP_BOUND'
        Options.NumberDuals = Inputs.NumberDuals;
        Options.Dual = rand(1,Inputs.NumberDuals);
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    case 'ENLP_BOUND'
        Options.NumberDuals = Inputs.NumberDuals;
        Options.Dual = rand(1,Inputs.NumberDuals);
        Options.NumberStates = Inputs.NumberStates;
        Options.State = rand(1,Inputs.NumberStates);
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    case 'ILP'
        Options.NumberDuals = Inputs.NumberDuals;
        Options.Dual = rand(1,Inputs.NumberDuals);
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    case 'CNLP'
        Options.NumberDuals = Inputs.NumberDuals;
        Options.Dual = rand(1,Inputs.NumberDuals);
        Options.NumberStates = Inputs.NumberStates;
        Options.State = rand(1,Inputs.NumberStates);
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    case 'CLP'
        Options.NumberDuals = Inputs.NumberDuals;
        Options.Dual = rand(1,Inputs.NumberDuals);
        Options.NumberStates = Inputs.NumberStates;
        Options.State = rand(1,Inputs.NumberStates);
        Options.NumberControls = Inputs.NumberControls;
        Options.Control = rand(1,Inputs.NumberControls);
    otherwise
        error(' Invalid Problem Type. See Users Manual. ');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            SET GENERAL OPTIONS                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setGeneralOptions(Options)
Options.MaxNumAlgorithmItr = 30;
Options.GradientTolerance = 1e-6;
Options.TrialStepTolerance = 1e-6;
Options.OptimalityTolerance = 1e-10;
Options.FeasibilityTolerance = 1e-10;
Options.ActualReductionTolerance = 1e-14;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            SET CONTAINER TYPE                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setContainerType(Options)
%   1. SERIAL_ARRAY
%   2. OMP_ARRAY
%   3. MPI_ARRAY
%   4. MPIx_ARRAY
Options.ContainerType = 'SERIAL_ARRAY';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            SET DISPLAY OPTIONS                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setDiagnosticsDisplayOption(Options)
%   1. ITERATION
%   2. FINAL
%   3. OFF
Options.OutputDataDisplayOption = 'ITERATION';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       BOUND CONSTRAINT OPTIONS                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setConstraintMethod(Options)
% Bound Constraint Options
%   1. FEASIBLE_DIR
%   2. ARMIJO_RULE_ALONG_FEASIBLE_DIR
%   3. PROJECTION_ALONG_FEASIBLE_DIR
%   4. ARMIJO_PROJECTION_ALONG_FEASIBLE_DIR
%   5. PROJECTION_ALONG_ARC
%   6. ARMIJO_RULE_ALONG_PROJECTED_ARC
Options.BoundConstraintMethod = 'PROJECTION_ALONG_FEASIBLE_DIR';
% General bound constraint method options
Options.MaxNumFeasibleItr = 2;
Options.BoundConstraintStepSize = 0.5;
Options.BoundConstraintContractionFactor = 0.5;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          TRUST REGION OPTIONS                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setTrustRegionMethod(Options)
% Trust-Region Options:
%   1. CAUCHY
%   2. DOGLEG
%   3. DOUBLE_DOGLEG
Options.TrustRegionMethod = 'DOUBLE_DOGLEG';
% General trust region options
Options.MaxTrustRegionRadius = 1e3;
Options.MinTrustRegionRadius = 1e-4;
Options.InitialTrustRegionRadius = 1e1;
Options.TrustRegionExpansionFactor = 2;
Options.TrustRegionContractionFactor = 0.25;
Options.MaxNumTrustRegionSubProblemItr = 1;
Options.MinActualOverPredictedReductionRatio = 0.1;
Options.MidActualOverPredictedReductionRatio = 0.25;
Options.MaxActualOverPredictedReductionRatio = 0.75;
Options.SetInitialTrustRegionRadiusToNormGrad = 'true';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           LINE SEARCH OPTIONS                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setLineSearchMethod(Options)
% Line Search Methods:
%   1. ARMIJO
%   2. GOLDSTEIN
%   3. CUBIC_INTRP
%   4. GOLDENSECTION
Options.LineSearchMethod = 'CUBIC_INTRP';
% General line search options
Options.MaxNumLineSearchItr = 5;
Options.LineSearchStagnationTolerance = 1e-7;
Options.LineSearchContractionFactor = 0.5;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       GRADIENT COMPUTATION OPTIONS                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setGradientComputationMethod(Options)
% Gradient Operator Options: 
%   1. FORWARD_DIFFERENCE
%   2. BACKWARD_DIFFERENCE
%   3. CENTRAL_DIFFERENCE
%   4. USER_DEFINED
%   5. PARALLEL_FORWARD_DIFFERENCE
%   6. PARALLEL_BACKWARD_DIFFERENCE
%   7. PARALLEL_CENTRAL_DIFFERENCE
Options.GradientComputationMethod = 'USER_DEFINED';
if (~strcmp(Options.GradientComputationMethod,'USER_DEFINED'))
    Options.FiniteDifferencePerturbations = [1e-5, 1e-6];
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           QUASI-NEWTON OPTIONS                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setQuasiNewtonMethod(Options)
% Inverse Hessian Operator Options: 
%   1. LBFGS
%   2. LDFP
%   3. LSR1
%   4. SR1
%   5. BFGS
%   6. USER_DEFINED
%   6. BARZILAI_BORWEIN
Options.QuasiNewtonStorage = 4;
Options.QuasiNewtonMethod = 'LSR1';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        HESSIAN COMPUTATION OPTIONS                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setHessianComputationMethod(Options)
% Hessian Operator Options: 
%   1. LBFGS
%   2. LDFP
%   3. LSR1
%   4. SR1
%   5. DFP
%   6. USER_DEFINED
%   7. USER_DEFINED_TYPE_CNP
%   8. BARZILAI_BORWEIN
%   9. FORWARD_FD
%   10. NUMERICALLY_INTEGRATED
Options.HessianComputationMethod = 'USER_DEFINED';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           NONLINEAR CG OPTIONS                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setNonLinearCGMethod(Options)
% Nonlinear CG Options
%   1. FLETCHER_REEVES
%   2. POLAK_RIBIERE
%   3. HESTENES_STIEFEL
%   4. CONJUGATE_DESCENT
%   5. HAGER_ZHANG
%   6. DAI_LIAO
%   7. DAI_YUAN
%   8. DAI_YUAN_HYBRID
%   9. PERRY_SHANNO
%  10. LIU_STOREY
%  11. DANIELS
Options.NonlinearCgMethod = 'PERRY_SHANNO';
if (strcmp(Options.NonlinearCgMethod,'DANIELS'))
    Options = setNumericalDifferentiationMethod(Options);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            KRYLOV SOLVER OPTIONS                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setKrylovSolver(Options)
% Krylov Solver Options
%   1. PCG
%   2. GMRES
%   3. PCGNR
%   4. PCGNE
%   5. PCR
%   6. PGCR
%   7. USER_DEFINED
Options.KrylovSolverMethod = 'PCG';
Options.FixTolerance = 1e0;
Options.RelativeTolerance = 1e0;
Options.MaxNumKrylovSolverItr = 100;
Options.RelativeToleranceExponential = 0.5;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       NUMERICAL INTEGRATION OPTIONS                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setNumericalDifferentiationMethod(Options)
% Preconditioner Types:
%   1. FORWARD_DIFFERENCE
%   2. BACKWARD_DIFFERENCE
%   3. CENTRAL_DIFFERENCE
%   4. SECOND_ORDER_FORWARD_DIFFERENCE
%   5. THIRD_ORDER_FORWARD_DIFFERENCE
%   6. THIRD_ORDER_BACKWARD_DIFFERENCE
Options.NumericalIntegrationMethod = 'CENTRAL_DIFFERENCE';
Options.NumericalIntegrationEpsilon = 1e-7;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     MMA ASYMPTOTE UPDATE RULE OPTIONS                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setAsymptoteUpdateRule(Options)
% Preconditioner Types:
%   1. FIXED_RULE
%   2. DYNAMIC_RULE
%   3. PRIMAL_SCALING_RULE
Options.AsymptoteUpdateRule = 'PRIMAL_SCALING_RULE';

switch Options.AsymptoteUpdateRule
    case 'FIXED_RULE'
        Options.LowerMoveLimitPenalty = 0.1;
        Options.UpperMoveLimitPenalty = 0.9;
        Options.FixedAsymptoteRulePenalty = 1.;
    case 'DYNAMIC_RULE'
        Options.LowerMoveLimitPenalty = 0.1;
        Options.UpperMoveLimitPenalty = 0.9;
        Options.DynamicAsymptoteRulePenalty = 0.7;
    case 'PRIMAL_SCALING_RULE'
        Options.PrimalScalingRulePenalty = 0.125;
        Options.UpperMoveLimitPrimalScaling = 2;
        Options.LowerMoveLimitPrimalScaling = 0.5;
        Options.UpperMoveLimitAsymptoteScaling = 0.99;
        Options.LowerMoveLimitAsymptoteScaling = 1.01;
    otherwise
        error(' Invalid MMA Asymptote Update Rule. See Users Manual. ');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         SQP PRECONDITIONER OPTIONS                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setSqpLeftPreconditioner(Options)
% Preconditioner Types:
%   1. NO_PREC
%   2. USER_DEFINED_PREC
%   3. FULL_SCHUR_PREC
%   4. INCOMPLETE_SCHUR_PREC
Options.LeftPreconditionerType = 'FULL_SCHUR_PREC';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            SET PROBLEM DATA                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setProblem(Inputs, Options) 
% 1) ULP - UNCONSTRAINED LINEAR PROGRAMMING
% 2) UNLP - UNCONSTRAINED NONLINEAR PROGRAMMING 
% 3) ELP - EQUALITY CONSTRAINED LINEAR PROGRAMMING
% 4) ENLP - EQUALITY CONSTRAINED NONLINEAR PROGRAMMING
% 5) LP_BOUND - BOUND CONSTRAINED LINEAR PROGRAMMING
% 6) NLP_BOUND - BOUND CONSTRAINED NONLINEAR PROGRAMMING
% 7) ELP_BOUND - BOUND+EQUALITY CONSTRAINED LINEAR PROGRAMMING
% 8) ENLP_BOUND - BOUND+EQUALITY CONSTRAINED NONLINEAR PROGRAMMING
% 9) CLP - GENERAL CONSTRAINED LINEAR PROGRAMMING (EQUALITY+INEQUALITY)
% 10) CLP - GENERAL CONSTRAINED NONLINEAR PROGRAMMING (EQUALITY+INEQUALITY)
% 11) CNLP - GENERAL CONSTRAINED NONLINEAR PROGRAMMING (EQUALITY+INEQUALITY)
% 12) ILP - INEQUALITY CONSTRAINED LINEAR PROGRAMMING
%
Options.ProblemType = Inputs.ProblemType;
Options = setInitialControl(Inputs, Options);

switch Options.ProblemType
    case 'ULP'
        Options.NumberControls = Inputs.NumberControls;
    case 'UNLP'
        Options = setInitialDual(Inputs, Options);
        Options.NumberDuals = Inputs.NumberDuals;
        Options.NumberControls = Inputs.NumberControls;
    case 'ELP'
        Options = setInitialDual(Inputs, Options);
        Options.NumberDuals = Inputs.NumberDuals;
        Options.NumberControls = Inputs.NumberControls;
    case 'ENLP'
        Options = setInitialDual(Inputs, Options);
        Options = setInitialState(Inputs, Options);
        Options.NumberDuals = Inputs.NumberDuals;
        Options.NumberStates = Inputs.NumberStates;
        Options.NumberControls = Inputs.NumberControls;
    case 'LP_BOUND'
        Options.NumberControls = Inputs.NumberControls;
        Options = setControlBounds(Inputs, Options);
        Options = setConstraintMethod(Options);
    case 'NLP_BOUND'
        Options = setInitialDual(Inputs, Options);
        Options.NumberDuals = Inputs.NumberDuals;
        Options.NumberControls = Inputs.NumberControls;
        Options = setControlBounds(Inputs, Options);
        Options = setConstraintMethod(Options);
    case 'ELP_BOUND'
        Options = setInitialDual(Inputs, Options);
        Options = setInitialState(Inputs, Options);
        Options.NumberDuals = Inputs.NumberDuals;
        Options.NumberControls = Inputs.NumberControls;
        Options = setControlBounds(Inputs, Options);
        Options = setConstraintMethod(Options);
    case 'ENLP_BOUND'
        Options = setInitialDual(Inputs, Options);
        Options = setInitialState(Inputs, Options);
        Options.NumberDuals = Inputs.NumberDuals;
        Options.NumberStates = Inputs.NumberStates;
        Options.NumberControls = Inputs.NumberControls;
        Options = setControlBounds(Inputs, Options);
        Options = setConstraintMethod(Options);
    case 'ILP'
        Options = setInitialDual(Inputs, Options);
        Options.NumberDuals = Inputs.NumberDuals;
        Options.NumberControls = Inputs.NumberControls;
        Options = setControlBounds(Inputs, Options);
    case 'CLP'
        Options = setInitialDual(Inputs, Options);
        Options = setInitialState(Inputs, Options);
        Options.NumberDuals = Inputs.NumberDuals;
        Options.NumberStates = Inputs.NumberStates;
        Options.NumberControls = Inputs.NumberControls;
        Options = setControlBounds(Inputs, Options);
    case 'CNLP'
        Options = setInitialDual(Inputs, Options);
        Options = setInitialState(Inputs, Options);
        Options.NumberDuals = Inputs.NumberDuals;
        Options.NumberStates = Inputs.NumberStates;
        Options.NumberControls = Inputs.NumberControls;
        Options = setControlBounds(Inputs, Options);
    otherwise
        error(' Invalid Problem Type. See Users Manual. ');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              SET OPERATORS                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Operators] = setOperators(Options) 

switch Options.ProblemType
    case 'ULP'
        Operators.ObjectiveFunction = objectiveFunction();
    case 'UNLP'
        Operators.ObjectiveFunction = objectiveFunction();
        Operators.EqualityConstraint = equalityConstraint();
    case 'ELP'
        Operators.ObjectiveFunction = objectiveFunction();
        Operators.EqualityConstraint = equalityConstraint();
    case 'ENLP'
        Operators.ObjectiveFunction = objectiveFunction();
        Operators.EqualityConstraint = equalityConstraint();
    case 'LP_BOUND'
        Operators.ObjectiveFunction = objectiveFunction();
    case 'NLP_BOUND'
        Operators.ObjectiveFunction = objectiveFunction();
        Operators.EqualityConstraint = equalityConstraint();
    case 'ELP_BOUND'
        Operators.ObjectiveFunction = objectiveFunction();
        Operators.EqualityConstraint = equalityConstraint();
    case 'ENLP_BOUND'
        Operators.ObjectiveFunction = objectiveFunction();
        Operators.EqualityConstraint = equalityConstraint();
    case 'ILP'
        Operators.ObjectiveFunction = objectiveFunction();
        Operators.InequalityConstraint = inequalityConstraint();
    case 'CNLP'
        Operators.ObjectiveFunction = objectiveFunction();
        Operators.EqualityConstraint = equalityConstraint();
        Operators.InequalityConstraint = inequalityConstraint();
    case 'CLP'
        Operators.ObjectiveFunction = objectiveFunction();
        Operators.EqualityConstraint = equalityConstraint();
        Operators.InequalityConstraint = inequalityConstraint();
    otherwise
        error(' Invalid Problem Type. See Users Manual. ');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      SET INITIAL OPTIMIZATION DATA                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setInitialDual(Inputs, Options)

if(length(Inputs.InitialDual) ~= Inputs.NumberDuals)
    error('Input NumberDuals (%d) IS NOT EQUAL to InitialDual Dim (%d)', ...
        Inputs.NumberDuals, length(Inputs.InitialDual));
else
    Options.Dual = Inputs.InitialDual;
end

end

function [Options] = setInitialState(Inputs, Options)

if(length(Inputs.InitialState) ~= Inputs.NumberStates)
    error('Input NumberStates (%d) IS NOT EQUAL to InitialState Dim (%d)', ...
        Inputs.NumberStates, length(Inputs.InitialState));
else
    Options.State = Inputs.InitialState;
end

end

function [Options] = setInitialControl(Inputs, Options)

if(length(Inputs.InitialControl) ~= Inputs.NumberControls)
    error('Input NumberControls (%d) IS NOT EQUAL to InitialControl Dim (%d)', ...
        Inputs.NumberControls, length(Inputs.InitialControl));
else
    Options.Control = Inputs.InitialControl;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         SET BOUNDS ON FIELD DATA                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setDualBounds(Inputs, Options)

if(length(Inputs.DualUpperBounds) ~= Inputs.NumberDuals)
    error(' Input NumberDuals (%d) IS NOT EQUAL to DualUpperBounds Dim (%d). ', ...
        Options.NumberDuals, length(Inputs.DualUpperBounds));
else
        Options.DualUpperBounds = Inputs.DualUpperBounds;
end

if(length(Inputs.DualLowerBounds) ~= Inputs.NumberDuals)
    error(' Input NumberDuals (%d) IS NOT EQUAL to DualLowerBounds Dim (%d). ', ...
        Options.NumberDuals, length(Inputs.DualLowerBounds));
else
        Options.DualLowerBounds = Inputs.DualLowerBounds;
end

end

function [Options] = setStateBounds(Inputs, Options)

if(length(Inputs.DualUpperBounds) ~= Inputs.NumberStates)
    error(' Input NumberStates (%d) IS NOT EQUAL to StateUpperBounds Dim (%d). ', ...
        Options.NumberStates, length(Inputs.StateUpperBounds));
else
        Options.StateUpperBounds = Inputs.StateUpperBounds;
end

if(length(Inputs.StateLowerBounds) ~= Inputs.NumberStates)
    error(' Input NumberStates (%d) IS NOT EQUAL to StateLowerBounds Dim (%d). ', ...
        Options.NumberStates, length(Inputs.StateLowerBounds));
else
        Options.StateLowerBounds = Inputs.StateLowerBounds;
end

end

function [Options] = setControlBounds(Inputs, Options)

if(length(Inputs.ControlUpperBounds) ~= Inputs.NumberControls)
    error(' Input NumberControls (%d) IS NOT EQUAL to ControlUpperBounds Dim (%d). ', ...
        Options.NumberControls, length(Inputs.ControlUpperBounds));
else
        Options.ControlUpperBounds = Inputs.ControlUpperBounds;
end

if(length(Inputs.ControlLowerBounds) ~= Inputs.NumberControls)
    error(' Input NumberControls (%d) IS NOT EQUAL to ControlLowerBounds Dim (%d). ', ...
        Options.NumberControls, length(Inputs.ControlLowerBounds));
else
        Options.ControlLowerBounds = Inputs.ControlLowerBounds;
end

end
