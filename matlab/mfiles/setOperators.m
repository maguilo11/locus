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
        Operators.InequalityConstraint = inequalityConstraint();
    otherwise
        error(' Invalid Problem Type. See Users Manual. ');
end

end