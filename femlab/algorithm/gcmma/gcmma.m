%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Globally Convergent Method of Moving Asymptotes (GCMMA)  
% DEVELOPER: MIGUEL A. AGUILO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
% 1. Redistributions of source code must retain the above copyright
% notice, this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution.
%
% 3. The name of Miguel A. Aguilo may not be used to endorse or promote 
% products derived from this software without specific prior written 
% permission.
%
% THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
% INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
% AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
% MIGUEL A. AGUILO BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
% OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = gcmma(objective, equality, inequality, control, ...
    control_lower_bound, control_upper_bound, a_coeff, d_coeff, ...
    c_coeff,max_outer_itr, residual_tolerance)

fprintf('\n*** Globally Convergent Method of Moving Asymptotes (GCMMA) ***\n');

% initialize algorithm data
Fold = 1e20;
number_controls = size(control,1);
sigma_old = zeros(number_controls,1);
active_set = zeros(number_controls,1);
control_minus_one = zeros(number_controls,1);
control_minus_two = zeros(number_controls,1);
number_inequalities = inequality.number_inequalities;
primal = zeros(number_controls+number_inequalities+1,1);
% Check that initial control is within the bounds
[primal(1:number_controls)] = ...
    checkInitialControl(control,control_lower_bound,control_upper_bound);
% Solve pde constraint
[state] = equality.solve(primal(1:number_controls));    % Update data
% Evaluate objective and inequality constraints
[function_values,~] = computeFunctionValues(state,...
    primal(1:number_controls),inequality,objective);
% Initialize auxiliary variables
[primal] = initAuxiliaryVariables(state, primal, inequality, a_coeff);
% outer optimization loop
iteration = 1;
grad_tolerance = 1e-3;
objective_stagnation_tol = 1e-8;
feasibility_tolerance = 1e-3;
control_stagnation_tolerance = 1e-8;
dual = zeros(number_inequalities,1);
feasibility_measure = ones(number_inequalities,1);
number_ccsa_functions = number_inequalities+1;
rho = ones(number_ccsa_functions,1);
rho_min = 1e-5*ones(number_ccsa_functions,1);
global GLB_INVP;

why = 'iteration';
GLB_INVP.iteration = iteration;
while(iteration <= max_outer_itr)
    % Assemble reduced objective and inequality gradients
    [function_grad] = computeFunctionGradients(state, ...
        primal(1:number_controls),active_set,inequality,objective);
    if(iteration == 1)
        initial_grad_norm = norm(function_grad(:,1),2);
    end
    [stop,~] = stoppingCriteriaSatisfied(primal,dual,...
        function_values,function_grad,residual_tolerance);
    control_stagnation_norm = ...
        norm(control_minus_one-primal(1:number_controls));
    objective_stagnation = abs(Fold - function_values(1));
    grad_norm = norm(function_grad(active_set~=1,1))  / initial_grad_norm;
    show(GLB_INVP.mesh.t, GLB_INVP.mesh.p, primal(1:number_controls));
    if(stop == true)
        why = 'residual';
        break;
    elseif(control_stagnation_norm < control_stagnation_tolerance)
        why = 'control_stagnation_norm';
        break;
    elseif((max(feasibility_measure) < feasibility_tolerance) && (grad_norm  < grad_tolerance))
        why = 'optimality_&_feasibility';
        break;
    elseif(objective_stagnation < objective_stagnation_tol)
        why = 'objective_stagnation';
                break;
    end
    % initialize control data
    [sigma]=computeSigmaParameters(iteration,primal(1:number_controls),...
        sigma_old,control_minus_one,control_minus_two,...
        control_lower_bound, control_upper_bound);
    sigma_old = sigma;
    % Update moving asymptotes
    lower_asymptote = primal(1:number_controls) - sigma;
    upper_asymptote = primal(1:number_controls) + sigma;
    % Solve dual problem
    control_minus_two = control_minus_one;
    control_minus_one = primal(1:number_controls);
    Fold = function_values(1);
    [function_values,feasibility_measure,state,primal,dual,rho,active_set,sub_itr] = ...
        subProblem(dual,objective,equality,inequality,primal,...
        function_values,function_grad,lower_asymptote,upper_asymptote,...
        sigma,rho,a_coeff,d_coeff,c_coeff,control_lower_bound,control_upper_bound);
    % update rho (globalization scaling) parameters
    rho = max(0.1*rho,rho_min);
    % next iteration
    iteration = iteration + 1;
    GLB_INVP.iteration = iteration;
end

[output] = saveOutput(dual,state,primal,iteration,...
    function_grad,function_values);
output.why = why;

end

function [output] = ...
    saveOutput(dual,state,primal,itr,function_grad,function_values)
number_inequalities = size(dual,1);
number_controls = size(function_grad,1);
% save output data
output.dual = dual;
output.state = state;
output.control = primal(1:number_controls);
output.aux_z = primal(size(primal,1));
output.aux_y = ...
    primal(number_controls+1:number_controls+number_inequalities);
output.iteration = itr;
output.function_grad = function_grad;
output.function_values = function_values;
end

function [function_values,feasibility_measure] = ...
    computeFunctionValues(state,control,inequality,objective)

number_inequalities = inequality.number_inequalities;
function_values = zeros(number_inequalities+1,1);
feasibility_measure = zeros(number_inequalities,1);
% Compute objective function value
[function_values(1)] = objective.evaluate(state,control);
% Compute inequality constraint values
for index=1:number_inequalities
    function_values(1+index) = inequality.evaluate(state,control,index) ...
        - inequality.value(index);
    feasibility_measure(index) = abs(function_values(1+index) / ...
        inequality.value(index));
end

end

function [function_grad] = ...
    computeFunctionGradients(state,control,active_set,inequality,objective)

number_controls = size(control,1);
number_inequalities = inequality.number_inequalities;
function_grad = zeros(number_controls,number_inequalities+1);
% Evaluate objective function gradient
[function_grad(:,1)] = objective.gradient(state,control);
% Evaluate inequality constraint gradient
for index=1:number_inequalities
    function_grad(:,1+index) = inequality.gradient(state,control,index);
end

end

function [new_function_values,feasibility_measure,state,primal,dual,rho,active_set,iteration] = ...
    subProblem(dual,objective, equality, inequality, ...
    current_primal, current_function_values, ...
    current_function_grad, lower_asymptote, ...
    upper_asymptote, sigma, rho,a_coeff,d_coeff,...
    c_coeff,control_lower_bound,control_upper_bound)
iteration = 1;
max_inner_itr = 5;
stopping_tolerance = 1e-8;
stagnation_tolerance = 1e-12;
old_primal = current_primal;
active_set = zeros(size(control_lower_bound));
old_function_values = current_function_values;
number_controls = size(upper_asymptote,1);
new_function_values = zeros(inequality.number_inequalities+1,1);
while(iteration <= max_inner_itr)
    % compute p and q coefficient vectors
    [p_coeff,q_coeff] = computeCoefficients(dual,sigma,rho,...
        current_function_grad);
    % Compute dual
    [primal,dual] = nonlinearCg(dual,current_function_values, ...
        current_function_grad,lower_asymptote,upper_asymptote,...
        sigma,rho,a_coeff,d_coeff,c_coeff,p_coeff,q_coeff,control_lower_bound,...
        control_upper_bound);
    % project primal control to lower/upper bound
    primal(1:number_controls) = ...
        max(primal(1:number_controls),control_lower_bound);
    primal(1:number_controls) = ...
        min(primal(1:number_controls),control_upper_bound);
    % Compute primal active set
    indices = (primal(1:number_controls) == control_lower_bound) | ...
        (primal(1:number_controls) == control_upper_bound);
    active_set(indices) = 1;
    % Solve new state solution
    [state] = equality.solve(primal(1:number_controls));
    % Evaluate objective and inequality constraints
    [new_function_values,feasibility_measure] = ...
        computeFunctionValues(state,primal(1:number_controls),...
        inequality,objective);
    % Compute globalization parameter rho
    [rho] = computeGlobalizationParameter(rho,sigma,...
        primal(1:number_controls),current_primal(1:number_controls),...
        new_function_values,current_function_values,...
        current_function_grad);
    % Check stopping criteria
    [stopping_criteria_met,norm_residual] = stoppingCriteriaSatisfied(primal,dual,...
        current_function_values,current_function_grad,stopping_tolerance);
    % Check for stagnation
    delta_primal = ...
        norm(old_primal(1:number_controls)-primal(1:number_controls));
    delta_fval = norm(old_function_values-new_function_values);
    if(stopping_criteria_met == true)
        break;
    elseif(delta_primal < stagnation_tolerance || ...
            delta_fval < stagnation_tolerance)
        break;
    end
    old_primal = primal;
    old_function_values = new_function_values;
    iteration = iteration + 1;
end

end

function [p_coeff,q_coeff] = computeCoefficients(dual,sigma,rho,...
    current_function_grad)
% compute p and q coefficient vectors
number_controls = size(sigma,1);
number_inequalities = size(dual,1);
p_coeff = zeros(number_controls,1+number_inequalities);
q_coeff = zeros(number_controls,1+number_inequalities);
% C++ implementation requires loop over controls (objective function term)
for i=1:number_inequalities+1
    p_coeff(:,i) = (sigma.^2).*max(0,current_function_grad(:,i)) ...
        + ((rho(i).*sigma)./4);
    q_coeff(:,i) = (sigma.^2).*max(0,-current_function_grad(:,i)) ...
        + ((rho(i).*sigma)./4);
end

end

function [rho] = computeGlobalizationParameter(rho,sigma,trial_control, ...
    current_control,trial_function_values,current_function_values,...
    current_function_grad)
% Evaluate each conservative convex separable approximation (CCSA)
num_inequalities = size(current_function_values,1) - 1;
ccsa_func = zeros(num_inequalities+1,1);
for i=1:num_inequalities+1
    % C++ implementation loops over controls
    delta_control = trial_control - current_control;
    w_function_evaluation = 0.5*sum(((delta_control).^2) ./ ...
        (sigma.^2 - delta_control.^2));
    numerator = ((sigma.^2).*current_function_grad(:,i).*delta_control) ...
        + (sigma.*abs(current_function_grad(:,i)).*(delta_control.^2));
    denominator = (sigma.^2) - (delta_control.^2);
    v_function_evaluation = current_function_values(i) + ...
        sum(numerator./denominator);
    % compute ccsa function
    ccsa_func(i) = v_function_evaluation + rho(i)*w_function_evaluation;
    % evaluate ratio between the actual and predicted reduction in the
    % objective function
    delta = (trial_function_values(i)-ccsa_func(i)) / ...
        w_function_evaluation;
    if(delta>0)
        rho(i) = min(10*rho(i),1.1*(rho(i)+delta));
    end
end

end

function [control] = checkInitialControl(control,lower_bound, upper_bound)

number_controls = size(control,1);
for i=1:number_controls
    % check that lower bound is met
    control(i) = max(lower_bound(i),control(i));
    % check that upper bound is met
    control(i) = min(upper_bound(i),control(i));
end

end

function [sigma] = ...
    computeSigmaParameters(iteration, control, sigma_old, ...
    control_minus_one, control_minus_two, ...
    control_lower_bound, control_upper_bound)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize sigma vector used during control update, e.g.
% primal_j = [primal_j-0.9*sigma_j,primal_j+0.9*sigma_j]
% where j=1...,n
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_controls = size(control,1);
sigma = zeros(num_controls,1);
if(iteration < 3)
    sigma = 0.5*(control_upper_bound - control_lower_bound);
else
    for i=1:num_controls
        % compute sigma forr this entry
        value = (control(i) - control_minus_one(i))*...
            (control_minus_one(i) - control_minus_two(i));
        if(value > 0)
            sigma(i) = 1.2*sigma_old(i);
        elseif((value < 0))
            sigma(i) = 0.4*sigma_old(i);
        else
            sigma(i) = sigma_old(i);
        end
    end
end

% check that lower bound is met
lower_sigma_bound = ...
    0.01*(control_upper_bound - control_lower_bound);
sigma = max(lower_sigma_bound,sigma);
% check that upper bound is met
upper_sigma_bound = ...
    10*(control_upper_bound - control_lower_bound);
sigma = min(upper_sigma_bound,sigma);

end

function [primal] = ...
    initAuxiliaryVariables(state, primal, inequality, a_coeff)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize auxiliary variables associated with the inequality constraints
% , e.g. primal = [X1,...,Xn,Y1,...,Ym,z], where n=number
% variables and m=number of inequality constraints. The auxiliary variables
% are given by Ym and z.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

number_primal = size(primal,1);
number_inequalities = inequality.number_inequalities;
number_controls = number_primal - (number_inequalities+1);
if(max(a_coeff(2:end)) == 0)
    for index=1:number_inequalities
        primal(number_controls+index) = ...
            -inequality.value(index) + ...
            inequality.evaluate(state,primal(1:number_controls),index);
    end
    primal(number_primal)=0;
else
    for index=1:number_inequalities
        possible_aux_z_values = zeros(number_inequalities,1);
        if(a_coeff(1+index) > 0)
            primal(number_controls+index)=0;
            value = -inequality.value(index) + ...
                inequality.evaluate(state,primal(1:number_controls),index);
            possible_aux_z_values(index) = ...
                max(0,value) / a_coeff(1+index);
        else
            value = -inequality.value(state,primal(1:number_controls),index) + ...
                inequality.evaluate(state,primal(1:number_controls),index);
            primal(number_controls+index) = max(0,value);
        end
    end
    primal(number_primal)=max(possible_aux_z_values);
end

end

function [primal,solution]=nonlinearCg(initial_dual,...
    current_function_values,current_function_grad,lower_asymptote,...
    upper_asymptote,sigma,rho,a_coeff,d_coeff,c_coeff,p_coeff,q_coeff,...
    control_lower_bound,control_upper_bound)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In numerical optimization, the nonlinear conjugate gradient method  %
% generalizes the conjugate gradient method to nonlinear optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize data structure
dataMng = [];
type = 'Hestenes_Stiefel';
dataMng.epsilon = 1e-6;
max_num_iterations = 50;
dataMng.rho = rho;
dataMng.sigma = sigma;
dataMng.a_coeff = a_coeff;
dataMng.d_coeff = d_coeff;
dataMng.c_coeff = c_coeff;
dataMng.p_coeff = p_coeff;
dataMng.q_coeff = q_coeff;
dataMng.lower_asymptote = lower_asymptote;
dataMng.upper_asymptote = upper_asymptote;
dataMng.control_lower_bound = control_lower_bound;
dataMng.control_upper_bound = control_upper_bound;
dataMng.current_function_grad = current_function_grad;
dataMng.current_function_values = current_function_values;
% Get objective function operators for dual problem
objective = dualObjectiveFunction;
% Compute initial objective function
dataMng.new_dual = 0;
dataMng.lower_bound = zeros(size(initial_dual));
dataMng.upper_bound = 1e16*ones(size(initial_dual));
% Evaluate initial dual objective function and primal vector
[dataMng.new_fval,primal] = ...
    objective.evaluate(dataMng.new_dual,dataMng);
% Compute initial gradient
[dataMng.new_gradient] = ...
    objective.gradient(dataMng.new_dual,primal,dataMng);
% Compute projected gradient
[dataMng.proj_gradient] = computeProjectedGradient(dataMng.new_dual,...
    dataMng.new_gradient,dataMng.lower_bound,dataMng.upper_bound);
% Set initial trial step to steepest descent
dataMng.new_steepest_descent = -dataMng.proj_gradient;
dataMng.new_trial_step = dataMng.new_steepest_descent;
% Store current data
[dataMng] = storeCurrentState(dataMng);
% Update primal vector via a line search routine
[dataMng,trial_dual,primal] = ...
    step(objective,dataMng.new_trial_step,dataMng);
dataMng.new_dual = trial_dual;
% Solve optimization problem
iteration = 1;
if(converged(dataMng) == true)
    solution=dataMng.new_dual;
    return;
end
while(iteration <= max_num_iterations)
    [dataMng.new_gradient] = ...
        objective.gradient(dataMng.new_dual,primal,dataMng);
    % Compute projected gradient
    [dataMng.proj_gradient] = computeProjectedGradient(dataMng.new_dual,...
        dataMng.new_gradient,dataMng.lower_bound,dataMng.upper_bound);
    % Compute steepest descent
    dataMng.new_steepest_descent = -dataMng.proj_gradient;
    % compute conjugate scaling
    [scaling] = computeScaling(dataMng,type);
    if(isfinite(scaling) == false)
        break;
    end
    dataMng.new_trial_step = dataMng.new_steepest_descent + ...
        (scaling*dataMng.old_trial_step);
    % Store current data
    [dataMng] = storeCurrentState(dataMng);
    % Line search step
    [dataMng,trial_dual,primal] = ...
        step(objective,dataMng.new_trial_step,dataMng);
    dataMng.new_dual = trial_dual;
    if(converged(dataMng) == true)
        break;
    end
    iteration = iteration + 1;
end
solution=dataMng.new_dual;
end

function [proj_gradient] = ...
    computeProjectedGradient(control,gradient,lower_bound,upper_bound)
step = -1;
proj_gradient = gradient;
trial_control = control + step.*gradient;
lower_binding_set = trial_control < lower_bound;
proj_gradient(lower_binding_set == 1) = 0.;
upper_binding_set = trial_control > upper_bound;
proj_gradient(upper_binding_set == 1) = 0.;
end

function [stop] = converged(dataMng)

stop=false;
norm_gradient = norm(dataMng.new_gradient);
norm_trial_step = norm(dataMng.new_trial_step);
if(norm_gradient < 1e-8)
    stop = true;
elseif(norm_trial_step < 1e-8)
    stop = true;
end

end

function [stop,norm_residual] = stoppingCriteriaSatisfied(primal,dual,...
    function_values,function_grad,tol)
stop = false;
num_inequalities = size(dual,1);
num_controls = size(primal,1) - num_inequalities - 1;
residual = zeros(2*num_controls + 2*num_inequalities,1);
inequality_constraint_dot_dual = function_grad(:,2:end)*dual;
% compute optimality residual
for j=1:num_controls
    value = function_grad(j,1) + inequality_constraint_dot_dual(j);
    residual(j) = residual(j) + (1+primal(j))*max(0,value);
    residual(num_controls+j) = residual(num_controls+j) + ...
        (1-primal(j))*max(0,-value);
end
% compute feasibility residual
for i=1:num_inequalities
    residual(2*num_controls+i) = ...
        max(0,function_values(1+i));
    residual(2*num_controls+num_inequalities+i) = ...
        dual(i)*max(0,-function_values(1+i));
end
norm_residual = (1/num_controls)*norm(residual,2);
if(norm_residual < tol)
    stop = true;
end

end

function [dataMng] = storeCurrentState(dataMng)

dataMng.old_fval = dataMng.new_fval;
dataMng.old_dual = dataMng.new_dual;
dataMng.old_gradient = dataMng.new_gradient;
dataMng.old_proj_gradient = dataMng.proj_gradient;
dataMng.old_trial_step = dataMng.new_trial_step;
dataMng.old_steepest_descent = dataMng.new_steepest_descent;

end

function [scaling] = computeScaling(dataMng,type)

switch(type)
    case 'Fletcher_Reeves'
        scaling = ...
            (dataMng.new_steepest_descent'*dataMng.new_steepest_descent) / ...
            (dataMng.old_steepest_descent'*dataMng.old_steepest_descent);
    case 'Polak_Ribiere'
        scaling = (dataMng.new_steepest_descent'*...
            (dataMng.new_steepest_descent-dataMng.old_steepest_descent)) ...
            / (dataMng.old_steepest_descent'*dataMng.old_steepest_descent);
    case 'Hestenes_Stiefel'
        scaling = (dataMng.new_steepest_descent'*...
            (dataMng.new_steepest_descent-dataMng.old_steepest_descent)) ...
            / (dataMng.old_trial_step'*(dataMng.new_steepest_descent - ...
            dataMng.old_steepest_descent));
    case 'Dai_Yuan'
        scaling = (dataMng.new_steepest_descent'*...
            dataMng.new_steepest_descent) / ...
            (dataMng.old_trial_step'*(dataMng.new_steepest_descent - ...
            dataMng.old_steepest_descent));
end

end

function [dataMng,trial_dual,primal] = step(objective,direction,dataMng)
%
% function [dataMng] = step(objective,dataMng)
%
% Armijo rule, polynomial linesearch
%
%
% Input: objective = objective function operators
%        dataMng = data manager. stores all the optimization problem
%                  related data structures
%
% Output: dataMng = updated data manager. updates objective function value
%                   and control data
%
% Requires: polymod.m
%
% linesearch parms
%
bhigh=.5; blow=1e-3;
%
%
alpha=1e-4;
num_line_search_iterations=1;
trial_dual = dataMng.new_dual;
max_num_line_search_itr = 10;
initial_fval = dataMng.new_fval;
lambda=min(1,100/(1+norm(dataMng.new_trial_step)));
current_lambda = lambda;
[trial_dual,~] = project(trial_dual,dataMng.lower_bound,...
    dataMng.upper_bound,lambda,direction);
dataMng.projected_trial_step = trial_dual - dataMng.new_dual;
[current_fval,primal] = objective.evaluate(trial_dual,dataMng);
dataMng.new_fval = current_fval;
initial_proj_trialStep_dot_gradient = ...
    dataMng.projected_trial_step'*dataMng.new_gradient;
%fgoal=initial_fval - alpha*lambda*initial_proj_trialStep_dot_gradient;
fgoal=initial_fval - alpha*lambda*initial_proj_trialStep_dot_gradient;
%
%       polynomial line search
%
while(current_fval > fgoal)
    if num_line_search_iterations==1
        lambda=polymod(initial_fval, initial_proj_trialStep_dot_gradient, ...
            current_lambda, current_fval, blow, bhigh);
    else
        lambda=polymod(initial_fval, initial_proj_trialStep_dot_gradient, ...
            current_lambda, current_fval, blow, bhigh, old_lambda, old_fval);
    end
    old_fval=current_fval;
    old_lambda=current_lambda;
    current_lambda=lambda;
    [trial_dual,~] = project(dataMng.new_dual,dataMng.lower_bound,...
        dataMng.upper_bound,lambda,direction);
    dataMng.projected_trial_step = trial_dual - dataMng.new_dual;
    [current_fval,primal] = objective.evaluate(trial_dual,dataMng);
    dataMng.new_fval = current_fval;
    if(num_line_search_iterations > max_num_line_search_itr)
       % disp(' Armijo error in steepest descent ')
        return;
    end
    fgoal=initial_fval - ...
        alpha*lambda*initial_proj_trialStep_dot_gradient;
    num_line_search_iterations=num_line_search_iterations+1;
end

end

function [lplus]=polymod(q0, qp0, lamc, qc, blow, bhigh, lamm, qm)
% function [lambda]=polymod(q0, qp0, qc, blow, bhigh, qm)
%
% Cubic/quadratic polynomial linesearch
%
% Finds minimizer lambda of the cubic polynomial q on the interval
% [blow * lamc, bhigh * lamc] such that
%
% q(0) = q0, q'(0) = qp0, q(lamc) = qc, q(lamm) = qm
%
% if data for a cubic is not available (first stepsize reduction) then
% q is the quadratic such that
%
% q(0) = q0, q'(0) = qp0, q(lamc) = qc
%
lleft=lamc*blow; lright=lamc*bhigh;
if nargin == 6
    %
    % quadratic model (temp hedge in case lamc is not 1)
    %
    lplus = - qp0/(2 * lamc*(qc - q0 - qp0) );
    if(lplus < lleft)
        lplus = lleft;
    end
    if(lplus > lright)
        lplus = lright;
    end
else
    %
    % cubic model
%     %
%     a=[lamc^2, lamc^3; lamm^2, lamm^3];
%     b=[qc; qm]-[q0 + qp0*lamc; q0 + qp0*lamm];
%     c=a\b;
%     lplus=(-c(1)+sqrt(c(1)*c(1) - 3 *c(2) *qp0))/(3*c(2));
%     if (lplus < lleft)
%         lplus = lleft;
%     end
%     if (lplus > lright)
%         lplus = lright;
%     end
    lplus = - qp0/(2 * lamc*(qc - q0 - qp0) );
    if(lplus < lleft)
        lplus = lleft;
    end
    if(lplus > lright)
        lplus = lright;
    end
end

end

function [projected_primal,active_set] = ...
    project(primal, lower_bound, upper_bound, scaling, direction)

number_primal = size(primal,1);
active_set = ones(number_primal,1);
projected_primal = primal + scaling*direction;
% C++ implementation requires loop over primal
for i=1:number_primal
    projected_primal(i) = max(projected_primal(i),lower_bound(i));
    active_set(i) = (projected_primal(i) > lower_bound(i));
    projected_primal(i) = min(projected_primal(i),upper_bound(i));
    active_set(i) = (projected_primal(i) < upper_bound(i));
    active_set(i) = ~((projected_primal(i) == upper_bound(i)) || ...
        (projected_primal(i) == lower_bound(i)));
end

end

function [primal,dual] = gradientProjection(initial_dual,...
    current_function_values,current_function_grad,lower_asymptote,...
    upper_asymptote,sigma,rho,a_coeff,d_coeff,c_coeff,p_coeff,q_coeff,...
    control_lower_bound,control_upper_bound)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In numerical optimization, the nonlinear conjugate gradient method  %
% generalizes the conjugate gradient method to nonlinear optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iteration = 1;
max_num_iterations = 100;
% Initialize data structure
dataMng = [];
dataMng.epsilon = 1e-6;
dataMng.rho = rho;
dataMng.sigma = sigma;
dataMng.a_coeff = a_coeff;
dataMng.d_coeff = d_coeff;
dataMng.c_coeff = c_coeff;
dataMng.p_coeff = p_coeff;
dataMng.q_coeff = q_coeff;
dataMng.lower_asymptote = lower_asymptote;
dataMng.upper_asymptote = upper_asymptote;
dataMng.control_lower_bound = control_lower_bound;
dataMng.control_upper_bound = control_upper_bound;
dataMng.current_function_grad = current_function_grad;
dataMng.current_function_values = current_function_values;
% Get objective function operators for dual problem
objective = dualObjectiveFunction;
% Compute initial objective function
dataMng.new_dual = 0;
dataMng.lower_bound = zeros(size(initial_dual));
dataMng.upper_bound = 1e16*ones(size(initial_dual));
% Evaluate initial dual objective function and primal vector
[dataMng.new_fval,primal] = ...
    objective.evaluate(dataMng.new_dual,dataMng);
% Compute initial gradient
[dataMng.new_gradient] = ...
    objective.gradient(dataMng.new_dual,primal,dataMng);   
% Compute projected gradient
trial_dual = dataMng.new_dual - dataMng.new_gradient;
dataMng.new_trial_step = trial_dual - dataMng.new_dual;
% Store current data
[dataMng] = storeCurrentState(dataMng);
% Update primal vector via a line search routine
[dataMng,trial_dual,primal] = ...
    step(objective,dataMng.new_trial_step,dataMng);
dataMng.new_dual = trial_dual;
% Solve optimization problem
if(converged(dataMng) == true)
    dual=dataMng.new_dual;
    return;
end
while(iteration <= max_num_iterations)
    [dataMng.new_gradient] = ...
        objective.gradient(dataMng.new_dual,primal,dataMng);
    % Compute projected gradient
    trial_dual = dataMng.new_dual - dataMng.new_gradient;
    dataMng.new_trial_step = trial_dual - dataMng.new_dual;
    % Store current data
    [dataMng] = storeCurrentState(dataMng);
    % Line search step
    [dataMng,trial_dual,primal] = ...
        step(objective,dataMng.new_trial_step,dataMng);
    dataMng.new_dual = trial_dual;
    if(converged(dataMng) == true)
        break;
    end
    iteration = iteration + 1;
end
dual=dataMng.new_dual;
end