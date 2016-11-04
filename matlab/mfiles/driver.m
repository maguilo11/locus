function [Output] = driver()
addpath('/Users/miguelaguilo/dotk/matlab/exe/');
addpath('/Users/miguelaguilo/dotk/matlab/mfiles/Rosenbrock/');
Input.ProblemType = 'LP_BOUND';
Input.SolutionType = 'NonLinearCG';
Input.NumberDuals = 0;
Input.NumberControls = 2;
Input.InitialControl = 2 * ones(1,Input.NumberControls);
Input.ControlLowerBounds = 0 * ones(1,Input.NumberControls);
Input.ControlUpperBounds = 5 * ones(1,Input.NumberControls);
[Options,Operators] = setOptions(Input);

switch Input.SolutionType
    case 'NonLinearCG'
        [Output] = mexDOTkNonLinearCG(Options, Operators);
    case 'QuasiNewton'
        [Output] = mexDOTkQuasiNewton(Options, Operators);
    case 'NewtonTypeLS'
        [Output] = mexDOTkNewtonTypeLS(Options, Operators);
    case 'NewtonTypeTR'
        [Output] = mexDOTkNewtonTypeTR(Options, Operators);
    case 'IxNewtonTypeLS'
        [Output] = mexDOTkInexactNewtonTypeLS(Options, Operators);
    case 'IxNewtonTypeTR'
        [Output] = mexDOTkInexactNewtonTypeTR(Options, Operators);
    case 'IxSqpTypeTR'
        [Output] = mexDOTkInexactSQPTypeTR(Options, Operators);
    case 'Diagnostics'
        checkOperators(Options, Operators);
    otherwise
        error(' Invalid Solution Type, See Users Manual. ');
end

end
