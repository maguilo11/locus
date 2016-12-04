function [Routines] = assemblyRoutines()
%% HIGH FIDELITY OPERATORS
Routines.objective=...
    @(Operators,OptDataMng,State,Primal)objective(Operators,OptDataMng,State,Primal);

Routines.objectiveGrad=...
    @(Operators,OptDataMng,State,Primal)objectiveGrad(Operators,OptDataMng,State,Primal);

Routines.inequality=...
    @(Operators,OptDataMng,State,Primal)inequality(Operators,OptDataMng,State,Primal);

Routines.inequalityGrad=...
    @(Operators,OptDataMng,State,Primal)inequalityGrad(Operators,OptDataMng,State,Primal);

Routines.hessian=...
    @(Operators,OptDataMng,State,Primal,Dual,vector)hessian(Operators,OptDataMng,State,Primal,Dual,vector);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value,State] = objective(Operators,OptDataMng,State,Primal)

Control.current = Primal.current(1:OptDataMng.nControls);
% Solve euqality constraint (PDE)
[State] = Operators.equality.solve(State,Control);
% Compute objective function
value = Operators.objective.evaluate(State,Control);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [residual] = inequality(Operators,OptDataMng,State,Primal)

nz = OptDataMng.nControls;
Slacks = Primal.current(1+nz:end);
Control.current = Primal.current(1:nz);
% Solve euqality constraint (PDE)
residual = Operators.inequality.residual(State,Control);
residual = residual - Slacks;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [grad,Dual] = objectiveGrad(Operators,OptDataMng,State,Primal)

Control.current = Primal.current(1:OptDataMng.nControls);
% Compute dual variables associated with the equality constraint
[rhs] = -1 .* (Operators.objective.firstDerivativeWrtState(State,Control));
[Dual] = Operators.equality.applyInverseAdjointJacobianWrtState(State,Control,rhs);

% Compute objective gradient
grad = Operators.objective.firstDerivativeWrtControl(State,Control);
% Gs_z_times_Dual = ...
%     Operators.equality.adjointFirstDerivativeWrtControl(State,Control,Dual);
% grad = F_z + Gs_z_times_Dual;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [grad,Dual] = inequalityGrad(Operators,OptDataMng,State,Primal)

Dual = 0;
Control.current = Primal.current(1:OptDataMng.nControls);
% Compute inequality gradient (**** Multiple constraints requires loop ****)
grad = Operators.inequality.firstDerivativeWrtControl(State,Control);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [kktTimesVector] = ...
    hessian(Operators,OptDataMng,State,Primal,Dual,perturbations)

% Gather respective perturbations
ns = OptDataMng.nSlacks;
nz = OptDataMng.nControls;
dz = perturbations(1:nz);
ds = perturbations(nz+1:end);
Control.current = Primal.current(1:nz);

% compute objective contribution to hessian times control perturbation operation (dz)
% Gauss-Newton Hessian
kktTimesVector = zeros(nz+ns,1);
kktTimesVector(1:nz) = Operators.objective.secondDerivativeWrtControlControl(State,Control,dz);
kktTimesVector(1:nz) = kktTimesVector(1:nz) + ...
    Operators.equality.secondDerivativeWrtControlControl(State,Control,Dual.objective,dz);

% compute perturbation in the direction of the inequality constraint lagrange multiplier
%   **** Multiple constraints will require a loop ****
Lambda = Primal.LagMult + ( Primal.penalty .* OptDataMng.currentHval );
kktTimesVector(1:nz) = kktTimesVector(1:nz) + ...
    ( ( Operators.inequality.secondDerivativeWrtControlControl(State,Control,dz) ) * Lambda );
dLambda = Primal.penalty * ( ( (Operators.inequality.firstDerivativeWrtControl(State,Control)')*dz ) );
kktTimesVector(1:nz) = kktTimesVector(1:nz) + ...
    ( ( Operators.inequality.firstDerivativeWrtControl(State,Primal) ) * dLambda );

% Add component L_zs*ds to first row block of kktTimesVector container
Hjac = Operators.inequality.firstDerivativeWrtControl(State,Primal);
kktTimesVector(1:nz) = kktTimesVector(1:nz) - (Primal.penalty * (Hjac * ds));

% Assemble remaining components of the KKT times vector container
kktTimesVector(nz+1:end) = -(Primal.penalty * (Hjac' * dz)) + ds;

end