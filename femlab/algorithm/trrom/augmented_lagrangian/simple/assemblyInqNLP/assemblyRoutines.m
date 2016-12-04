function [Routines] = assemblyRoutines()
%% HIGH FIDELITY OPERATORS
Routines.objective=...
    @(Operators,State,Primal)objective(Operators,State,Primal);

Routines.objectiveGrad=...
    @(Operators,State,Control)objectiveGrad(Operators,State,Control);

Routines.inequality=...
    @(Operators,OptDataMng,State,Primal)inequality(Operators,OptDataMng,State,Primal);

Routines.inequalityGrad=...
    @(Operators,State,Control)inequalityGrad(Operators,State,Control);

Routines.hessian=...
    @(state,control,rhs)hessian(state,control,rhs);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [value,State] = objective(Operators,State,Primal)
Control.current = Primal.current(1:OptDataMng.nControls);
% Solve euqality constraint (PDE)
[State] = Operators.equality.solve(State,Control);
% Compute objective function
value = Operators.objective.evaluate(State,Control);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [residual] = inequality(Operators,OptDataMng,State,Primal)
Slacks.current = ...
    Primal.current(1+OptDataMng.nControls:OptDataMng.nControls+OptDataMng.nSlacks);
Control.current = Primal.current(1:OptDataMng.nControls);
% Solve euqality constraint (PDE)
residual = ...
    Operators.inequality.evaluate(State,Control) - Operators.inequality.value();
residual = residual - Slacks;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [grad,Dual] = objectiveGrad(Operators,State,Control)

% Compute dual variables associated with the equality constraint
[rhs] = -1 .* (Operators.objective.firstDerivativeWrtState(State,Control));
[Dual.objective] = Operators.equality.applyInverseAdjointJacobianWrtState(State,Control,rhs);

% Compute objective gradient
F_z = Operators.objective.firstDerivativeWrtControl(State,Control);
Gs_z_times_Dual = ...
    Operators.equality.adjointFirstDerivativeWrtControl(State,Control,Dual.objective);
grad = F_z + Gs_z_times_Dual;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [grad,Dual] = inequalityGrad(Operators,State,Control)

% Compute dual variables associated with the inequality constraint
[rhs] = -1 .* (Operators.inequality.firstDerivativeWrtState(State,Control));
[Dual.inequality] = ...
    Operators.equality.applyInverseAdjointJacobianWrtState(State,Control,rhs);

% Compute inequality gradient (**** Multiple constraints requires loop ****)
H_z = Operators.inequality.firstDerivativeWrtControl(State,Control);
Gs_z_Times_Dual = ...
    Operators.equality.adjointFirstDerivativeWrtControl(State,Control,Dual.inequality);
grad = H_z + Gs_z_Times_Dual;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [kktTimesVector] = ...
    hessian(Operators,OptDataMng,State,Primal,Dual,perturbations)

% Gather respective perturbations
ns = OptDataMng.nSlacks;
nz = OptDataMng.nControls;
dz = perturbations(1:nz);
ds = perturbations(nz+1:nz+ns);
dv = perturbations(nz+ns+1:end);

% compute perturbation in the direction of the state vector
Control.current = Primal.current(1:nz);
rhs = -1.*(Operators.equality.firstDerivativeWrtControl(State,Control,dz));
[du] = Operators.equality.applyInverseJacobianWrtState(State,Control,rhs);

% compute perturbation in the direction of the equality constraint lagrange multiplier
rhs = Operators.equality.secondDerivativeWrtStateControl(State,Control,Dual.objective,dz) + ...
    Operators.equality.secondDerivativeWrtStateState(State,Control,Dual.objective,du);
rhs = rhs + Operators.objective.secondDerivativeWrtStateControl(State,Control,dz) + ...
    Operators.objective.secondDerivativeWrtStateState(State,Control,du);
rhs = -1 .* rhs;
[dLamda_obj] = Operators.equality.applyInverseJacobianWrtState(State,Control,rhs);

% compute objective contribution to hessian times control perturbation operation (dz)
hess_dz = Operators.objective.secondDerivativeWrtControlControl(State,Control,dz) + ...
    Operators.objective.secondDerivativeWrtControlState(State,Control,du);
hess_dz = hess_dz + Operators.equality.secondDerivativeWrtControlControl(State,Control,Dual.objective,dz) + ...
    Operators.equality.secondDerivativeWrtControlState(State,Control,Dual.objective,du);
hess_dz = hess_dz + Operators.equality.adjointFirstDerivativeWrtControl(State,Control,dLamda_obj);

% Add objective function contribution to the first row block of kktTimesVector
ndof = nz + 2*ns;
kktTimesVector = zeros(ndof,1);
kktTimesVector(1:nz) = hess_dz;

% compute perturbation in the direction of the inequality constraint lagrange multiplier
%   **** Multiple constraints will require loop and multiple solves ****
rhs = Operators.equality.secondDerivativeWrtStateControl(State,Control,Dual.inequality,dz) + ...
    Operators.equality.secondDerivativeWrtStateState(State,Control,Dual.inequality,du);
rhs = rhs + Operators.inequality.secondDerivativeWrtStateControl(State,Control,dz) + ...
    Operators.inequality.secondDerivativeWrtStateState(State,Control,du);
rhs = -1 .* rhs;
[dLambda_iq] = Operators.equality.applyInverseJacobianWrtState(State,Control,rhs);
LagMult = Primal.current(1+nz+ns:end); 
Lambda = LagMult + ( Dual.penalty * OptDataMng.currentHval );
kktTimesVector(1:nz) = kktTimesVector(1:nz) + ...
    ( ( Operators.inequality.secondDerivativeWrtControlControl(State,Control,dz) + ...
    Operators.inequality.secondDerivativeWrtControlState(State,Control,du) ) * Lambda );
kktTimesVector(1:nz) = kktTimesVector(1:nz) + ...
    ( ( Operators.equality.secondDerivativeWrtControlControl(State,Control,Dual.inequality,dz) + ...
    Operators.equality.secondDerivativeWrtControlState(State,Control,Dual.inequality,du) ) * Lambda );
dLambda = Dual.penalty * ( ( Operators.inequality.firstDerivativeWrtControl(State,Control)*dz + ...
    Operators.inequality.firstDerivativeWrtState(State,Control)*du ) );
kktTimesVector(1:nz) = kktTimesVector(1:nz) + ...
    ( ( Operators.inequality.firstDerivativeWrtControl(State,Primal) + ...
    Operators.equality.adjointFirstDerivativeWrtControl(State,Control,Dual.inequality) ) * dLambda );
kktTimesVector(1:nz) = kktTimesVector(1:nz) + ...
    ( Operators.equality.adjointFirstDerivativeWrtControl(State,Control,dLambda_iq) * dLambda );

% Add components L_zs*ds and L_zv*dv to first row block of kktTimesVector
% container
Hjac = Operators.inequality.firstDerivativeWrtControl(State,Primal) + ...
    Operators.equality.adjointFirstDerivativeWrtControl(State,Control,Dual.inequality);
kktTimesVector(1:nz) = kktTimesVector(1:nz) - ...
    (Dual.penalty * Hjac' * ds) + (Hjac' * dv);

% Assemble remaining components of the KKT times vector container
kktTimesVector(nz+1:nz+ns) = -(Dual.penalty * (Hjac * dz)) + ds - dv;
kktTimesVector(nz+ns:end) = (Hjac * dz) - ds;

end