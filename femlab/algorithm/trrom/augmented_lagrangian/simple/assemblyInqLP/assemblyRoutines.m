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

State.current = 0;
Control.current = Primal.current(1:OptDataMng.nControls);
% Compute objective function
value = Operators.objective.evaluate(State,Control);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [residual] = inequality(Operators,OptDataMng,State,Primal)

nz = OptDataMng.nControls;
Slacks = Primal.current(1+nz:end);
Control.current = Primal.current(1:nz);
% Compute residual
residual = Operators.inequality.residual(State,Control);
if(size(Slacks,1)>0)
    residual = residual - Slacks;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [grad,Dual] = objectiveGrad(Operators,OptDataMng,State,Primal)

Dual = 0;
Control.current = Primal.current(1:OptDataMng.nControls);
% Compute objective gradient
grad = Operators.objective.firstDerivativeWrtControl(State,Control);

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
if(ns>0)
    
    % compute objective contribution to hessian times control perturbation operation (dz)
    % Gauss-Newton Hessian
    kktTimesVector = zeros(nz+ns,1);
    Control.current = Primal.current(1:nz);
    F_hess = ...
        Operators.objective.secondDerivativeWrtControlControl(State,Control,dz);
    
    % apply perturbation to inequality constraint Hessian
    %   **** Multiple constraints will require a loop ****
    H_zz_times_dz = ...
        Operators.inequality.secondDerivativeWrtControlControl(State,Control,dz);
    Lambda = Dual.inequality - Primal.penalty*OptDataMng.currentHval;
    
    
    H_jac = Operators.inequality.firstDerivativeWrtControl(State,Primal);
    H_jac_times_dz = H_jac' * dz;
    H_jac_term = H_jac' * H_jac_times_dz;
    
    % Add component L_zs*ds to first row block of kktTimesVector container
    ds = perturbations(nz+1:end);
    Lzs = H_jac' * ds;
    penalty_x_H_jac_times_dz = Primal.penalty*(H_jac_term);
    kktTimesVector(1:nz) = F_hess + H_hess + ...
        penalty_x_H_jac_times_dz - (Primal.penalty * Lzs);
    
    % Assemble remaining components of the KKT times vector container
    kktTimesVector(nz+1:end) = -penalty_x_H_jac_times_dz + ds;
else
    % compute objective contribution to hessian times control perturbation operation (dz)
    Control.current = Primal.current(1:nz);
    F_hess = Operators.objective.secondDerivativeWrtControlControl(State,Control,dz);
    
    % apply perturbation to inequality constraint Hessian
    %   **** Multiple constraints will require a loop ****   
    H_zz_times_dz = ...
        Operators.inequality.secondDerivativeWrtControlControl(State,Control,dz);
    Lambda = Primal.LagMult - Primal.penalty*OptDataMng.currentHval;
    H_hess = H_zz_times_dz * Lambda;

    kktTimesVector = F_hess - H_hess;
end

end