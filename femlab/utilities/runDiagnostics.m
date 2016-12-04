function runDiagnostics(problem,InputData)
% Function that calls SOL diagnostics tools and checks problem operators
% In:
%   problem = flags that indicates the type of optimization problem solved
%   InputData = Struct that stores each optimization field, i.e. state and
%               controls
% AUTHOR: 
%       Miguel A. Aguilo (maguilo@sandia.gov)
% NOTES:
%       Derivative Flags:
%           U (first derivative with respect to (w.r.t.) states)
%           Z (first derivative w.r.t. controls)
%           UU (derivative of the (first derivative w.r.t. states) w.r.t. states)
%           ZZ (derivative of the (first derivative w.r.t. controls) w.r.t. controls)
%           UZ (derivative of the (first derivative w.r.t. states) w.r.t. controls)
%           ZU (derivative of the (first derivative w.r.t. controls) w.r.t. states)
%
fprintf(1,' PERFORM PROBLEM DIAGNOSTICS \n\n');
switch problem
    case 'Unconstrained'
        [Interface] = getDOTkDiagnosticsInterface('ObjectiveFunction');
        fprintf(1,' Check first derivative of the unconstrained objective function (F_z, z=control)\n');
        mxCheckFirstDerivativeUnconstrainedObjectiveFunction(InputData,Interface);
        fprintf(1,' Check second derivaitve of the unconstrained objective function (F_zz)\n');
        mxCheckSecondDerivativeUnconstrainedObjectiveFunction(InputData,Interface);
    case {'Constrained', 'EqualityConstrained'}
        % Objective Function 
        [Interface] = getDOTkDiagnosticsInterface('ObjectiveFunction');   
        fprintf(1,' Check first derivative of the constrained objective function (F_u, u=state)\n');
        mxCheckFirstDerivativeConstrainedObjectiveFunction(InputData,Interface,'U');
        fprintf(1,' Check first derivative of the constrained objective function with respect to the controls (F_z, z=control)\n');
        mxCheckFirstDerivativeConstrainedObjectiveFunction(InputData,Interface,'Z');
        fprintf(1,' Check second derivaitve of the constrained objective function (F_uu)\n');
        mxCheckSecondDerivativeConstrainedObjectiveFunction(InputData,Interface,'UU');
        fprintf(1,' Check second derivaitve of the constrained objective function (F_zz)\n');
        mxCheckSecondDerivativeConstrainedObjectiveFunction(InputData,Interface,'ZZ');
        fprintf(1,' Check second derivaitve of the constrained objective function (F_uz)\n');
        mxCheckSecondDerivativeConstrainedObjectiveFunction(InputData,Interface,'UZ');
        fprintf(1,' Check second derivaitve of the constrained objective function (F_zu)\n');
        mxCheckSecondDerivativeConstrainedObjectiveFunction(InputData,Interface,'ZU');
        % Equality Constraint
        [Interface] = getDOTkDiagnosticsInterface('EqualityConstraint');
        fprintf(1,' Check first derivative of the equality constraint (C_u, u=state)\n');
        mxCheckFirstDerivativeEqualityConstraint(InputData,Interface,'U');
        fprintf(1,' Check first derivative of the equality constraint (C_z, z=control)\n');
        mxCheckFirstDerivativeEqualityConstraint(InputData,Interface,'Z');
        fprintf(1,' Check adjoint of first derivative of the equality constraint (adjC_u, u=state)\n');
        mxCheckAdjointFirstDerivativeEqualityConstraint(InputData,Interface,'U');
        fprintf(1,' Check adjoint of first derivative of the equality constraint (adjC_z, z=control)\n');
        mxCheckAdjointFirstDerivativeEqualityConstraint(InputData,Interface,'Z');
        fprintf(1,' Check second derivative of the equality constraint (C_uu)\n');
        mxCheckAdjointSecondDerivativeEqualityConstraint(InputData,Interface,'UU');
        fprintf(1,' Check second derivative of the equality constraint (C_zz)\n');
        mxCheckAdjointSecondDerivativeEqualityConstraint(InputData,Interface,'ZZ');
        fprintf(1,' Check second derivative of the equality constraint (C_uz)\n');
        mxCheckAdjointSecondDerivativeEqualityConstraint(InputData,Interface,'UZ');
        fprintf(1,' Check second derivative of the equality constraint (C_zu)\n');
        mxCheckAdjointSecondDerivativeEqualityConstraint(InputData,Interface,'ZU');
    otherwise
        'Invalid problem type. Options are: Unconstrained,  EqualityConstrained or Constrained';
end
end