function checkOperators(Options, Operators)
% Check First Derivative
switch Options.ProblemType
    case 'ULP'
        mexDOTkCheckFirstDerivativeTypeLP(Options, Operators);
    case 'UNLP'
        mexDOTkCheckFirstDerivativeTypeNLP(Options, Operators);
    case 'ELP'
        mexDOTkCheckFirstDerivativeTypeLP(Options, Operators);
    case 'ENLP'
        mexDOTkCheckFirstDerivativeTypeNLP(Options, Operators);
    case 'LP_BOUND'
        mexDOTkCheckFirstDerivativeTypeLP(Options, Operators);
    case 'NLP_BOUND'
        mexDOTkCheckFirstDerivativeTypeNLP(Options, Operators);
    case 'ELP_BOUND'
        mexDOTkCheckFirstDerivativeTypeLP(Options, Operators);
    case 'ENLP_BOUND'
        mexDOTkCheckFirstDerivativeTypeNLP(Options, Operators);
    case 'CLP'
        mexDOTkCheckFirstDerivativeTypeLP(Options, Operators);
    case 'CNLP'
        mexDOTkCheckFirstDerivativeTypeNLP(Options, Operators);
    case 'ILP'
        mexDOTkCheckFirstDerivativeTypeLP(Options, Operators);
    otherwise
        error(' Invalid Problem Type. See Users Manual. ');
end
% Check Second Derivatives
if(Options.checkSecondDerivative == true)

    switch Options.ProblemType
    case 'ULP'
        mexDOTkCheckSecondDerivativeTypeLP(Options, Operators);
    case 'UNLP'
        mexDOTkCheckSecondDerivativeTypeNLP(Options, Operators);
    case 'ELP'
        mexDOTkCheckSecondDerivativeTypeLP(Options, Operators);
    case 'ENLP'
        mexDOTkCheckSecondDerivativeTypeNLP(Options, Operators);
    case 'LP_BOUND'
        mexDOTkCheckSecondDerivativeTypeLP(Options, Operators);
    case 'NLP_BOUND'
        mexDOTkCheckSecondDerivativeTypeNLP(Options, Operators);
    case 'ELP_BOUND'
        mexDOTkCheckSecondDerivativeTypeLP(Options, Operators);
    case 'ENLP_BOUND'
        mexDOTkCheckSecondDerivativeTypeNLP(Options, Operators);
    case 'CLP'
        mexDOTkCheckSecondDerivativeTypeLP(Options, Operators);
    case 'CNLP'
        mexDOTkCheckSecondDerivativeTypeNLP(Options, Operators);
    otherwise
        error(' Invalid Problem Type. See Users Manual. ');
    end

end
        
end