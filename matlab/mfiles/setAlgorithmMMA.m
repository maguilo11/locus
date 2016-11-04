function [Options, Operators] = setAlgorithmMMA(Inputs)
Options = [];
% Set Solution Type
Options.SolutionType = 'MMA';
% Set Container Type
Options.ContainerType = 'SERIAL_ARRAY';
% Set Problem Data Structures
Options = setProblemDataStruc(Inputs, Options);
% Set MMA Algorithm Options
Options = setOptionsMMA(Options);
% Set Problem Operators 
[Operators] = setOperators(Inputs);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            SET GENERAL OPTIONS                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setOptionsMMA(Options)

% 1) POLAK_RIBIERE; 2) FLETCHER_REEVES; 3) HESTENES_STIEFEL; 
% 4) CONJUGATE_DESCENT; 5) DAI_YUAN; 6) LIU_STOREY
Options.DualSolverType = 'NLCG';
Options.DualSolverTypeNLCG = 'POLAK_RIBIERE';
Options.MaxNumLineSearchItr = 5;
Options.LineSearchStepLowerBound = 1e-3;
Options.LineSearchStepUpperBound = 0.5;
Options.DualSolverMaxNumberIterations = 10;
Options.DualSolverGradientTolerance = 1e-8;
Options.DualSolverTrialStepTolerance = 1e-8;
Options.DualObjectiveEpsilonParameter = 1e-6;
Options.DualObjectiveTrialControlBoundScaling = 0.5;
Options.DualSolverObjectiveStagnationTolerance = 1e-8;

Options.MaxNumAlgorithmItr = 50;
Options.ResidualTolerance = 1e-4;
Options.GradientTolerance = 1e-3;
Options.OptimalityTolerance = 1e-4;
Options.FeasibilityTolerance = 1e-4;
Options.ControlStagnationTolerance = 1e-3;

Options.MovingAsymptoteUpperBoundScale = 10;
Options.MovingAsymptoteLowerBoundScale = 1e-1;
Options.MovingAsymptoteExpansionParameter = 1.2;
Options.MovingAsymptoteContractionParameter = 0.4;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            SET PROBLEM DATA                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setProblemDataStruc(Inputs, Options) 
% 1) CLP - GENERAL CONSTRAINED NONLINEAR PROGRAMMING (EQUALITY+INEQUALITY)
% 2) CNLP - GENERAL CONSTRAINED NONLINEAR PROGRAMMING (EQUALITY+INEQUALITY)
%
Options.ProblemType = Inputs.ProblemType;
Options = setInitialControl(Inputs, Options);

switch Options.ProblemType
    case 'CLP'
        Options = setInitialDual(Inputs, Options);
        Options = setInitialState(Inputs, Options);
        Options.NumberDuals = Inputs.NumberDuals;
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
%                      SET INITIAL OPTIMIZATION DATA                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Options] = setInitialDual(Inputs, Options)

Options.Dual = zeros(Inputs.NumberDuals,1);

end

function [Options] = setInitialState(Inputs, Options)

Options.State = zeros(Inputs.NumberStates,1);

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
