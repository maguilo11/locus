function [Operators] = dualObjectiveFunction()
Operators.evaluate=@(dual,dataMng)evaluate(dual,dataMng);
Operators.gradient=@(dual,primal,dataMng)gradient(dual,primal,dataMng);
end

% Evaluate dual objective function
function [output,primal] = evaluate(dual,dataMng)

num_inequalities = size(dual,1);
num_controls = size(dataMng.lower_asymptote,1);
% compute trial primal based on current dual value
[primal] = computeTrialPrimal(dual,dataMng);
num_primal = size(primal,1);
% compute sensitivity with respect to controls (% C++ implementation
% requires loop)
p_coeff_dot_dual = dataMng.p_coeff(:,2:end)*dual;
q_coeff_dot_dual = dataMng.q_coeff(:,2:end)*dual;
p_coeff_sum_term = (dataMng.p_coeff(:,1) + p_coeff_dot_dual) ...
    ./ (dataMng.upper_asymptote - primal(1:num_controls));
q_coeff_sum_term = (dataMng.q_coeff(:,1) + q_coeff_dot_dual) ...
    ./ (primal(1:num_controls) - dataMng.lower_asymptote);
sensitivity_wrt_control = p_coeff_sum_term - q_coeff_sum_term;
% project primal to feasible set
[primal] = projectPrimal(primal,sensitivity_wrt_control,dataMng);
% evaluate objective function
objective_r_term = dataMng.current_function_values(1) - ...
    sum((dataMng.p_coeff(:,1)+dataMng.q_coeff(:,1)) ./ ...
    dataMng.sigma);
objective_term = objective_r_term + ...
    primal(num_primal)*dataMng.a_coeff(1) + ...
    dataMng.epsilon*primal(num_primal)*primal(num_primal);
% summation over inequalities terms (C++ implementation requires loop)
first_elem = num_controls+1;
last_elem = num_controls+num_inequalities;
sum_over_aux_y_vars = sum(dataMng.c_coeff(1:end).*...
    primal(first_elem:last_elem) + (0.5*(dataMng.d_coeff(1:end).*...
    primal(first_elem:last_elem).*primal(first_elem:last_elem))));
inequality_r_term = dataMng.current_function_values(2:end) - ...
    sum((dataMng.p_coeff(:,2:end)+dataMng.q_coeff(:,2:end)) ./ ...
    dataMng.sigma);
dual_dot_inequality_r_term = dual'*inequality_r_term;
dual_dot_aux_variables = dual'*primal(first_elem:last_elem) + ...
    ((dual'*dataMng.a_coeff(2:end))*primal(num_primal));

mma_term = sum(p_coeff_sum_term + q_coeff_sum_term);
% add all terms
output = -1*(objective_term + sum_over_aux_y_vars - ...
    dual_dot_aux_variables + mma_term + dual_dot_inequality_r_term);
end
% Evalaute dual gradient operator
function [output] = gradient(dual,primal,dataMng)

output = zeros(size(dual));
num_primal = size(primal,1);
num_inequalities = size(output,1);
num_controls = num_primal - num_inequalities - 1;

for i=1:num_inequalities
    % compute r coefficient term
    r_term = dataMng.current_function_values(1+i) - ...
        sum((dataMng.p_coeff(:,1+i)+dataMng.q_coeff(:,1+i)) ./ ...
        dataMng.sigma);
    % C++ implementation requires loop
    p_coeff_sum_term = dataMng.p_coeff(:,1+i) ./ ...
        (dataMng.upper_asymptote - primal(1:num_controls));
    q_coeff_sum_term = dataMng.q_coeff(:,1+i) ./ ...
        (primal(1:num_controls) - dataMng.lower_asymptote);
    mma_term = sum(p_coeff_sum_term + q_coeff_sum_term);
    % add contribution
    output(i) = -primal(num_controls+i) - (dataMng.a_coeff(1+i) * ...
        primal(num_primal)) + mma_term + r_term;
end
output = -1.*output;

end

function [primal] = computeTrialPrimal(trial_dual,dataMng)
%
% function [primal] = computeTrialPrimal(trial_dual,lower_asymptote,...
%    upper_asymptote,rho,sigma,new_function_values,current_function_grad)
%
% Update primal vector based on trial dual variables computed during dual
% solve
%
number_inequalities = size(trial_dual,1);
number_controls = size(dataMng.lower_asymptote,1);
dual_dot_Pcoeff = dataMng.p_coeff(:,2:end)*trial_dual;
dual_dot_Qcoeff = dataMng.q_coeff(:,2:end)*trial_dual;
% C++ implementation has loop over control (possibly loop over 
% inequalities since dot product of p and q coefficients are needed)
a_term = dataMng.p_coeff(:,1)+dual_dot_Pcoeff;
b_term = dataMng.q_coeff(:,1)+dual_dot_Qcoeff;
primal = zeros(number_inequalities + number_controls + 1,1);
primal(1:number_controls) = ((dataMng.lower_asymptote.*(a_term.^0.5)) + ...
    (dataMng.upper_asymptote.*(b_term.^0.5))) ./ ((a_term.^0.5)+(b_term.^0.5));
% Update auxiliary variables Y_i
primal(number_controls+1:number_controls+number_inequalities)=...
    (trial_dual - dataMng.c_coeff) ./ dataMng.d_coeff;
% Update auxiliary variable Z 
primal(number_controls+number_inequalities+1) = ...
    (trial_dual'*dataMng.a_coeff(2:end) - dataMng.a_coeff(1)) ...
    / (2*dataMng.epsilon);
end

% project primal
function [primal] = projectPrimal(primal,sensitivity_wrt_control,dataMng)

% project control variables to feasible set
scaling = 0.9;
number_primals = size(primal,1);
number_controls = size(dataMng.sigma,1);
number_inequalities = number_primals - number_controls - 1;
% Compute lower and upper bounds
lower_bound = primal(1:number_controls) - (scaling.*dataMng.sigma);
upper_bound = primal(1:number_controls) + (scaling.*dataMng.sigma);
% C++ implementation requires loop over controls
% for index=1:number_controls
%     if(sensitivity_wrt_control(index) >= 0)
%         primal(index) = lower_bound(index);
%     elseif(sensitivity_wrt_control(index) <= 0)
%         primal(index) = upper_bound(index);
%     end
% end
primal(1:number_controls) = max(primal(1:number_controls),lower_bound);
primal(1:number_controls) = min(primal(1:number_controls),upper_bound);

% project primal control to lower/upper bound
%primal(1:number_controls) = ...
%    max(primal(1:number_controls),dataMng.control_lower_bound);
%primal(1:number_controls) = ...
%    min(primal(1:number_controls),dataMng.control_upper_bound);

% project auxiliary variables Y_j (C++ implementation requries loop)
primal(number_controls+1:number_primals-1) = ...
    max(primal(number_controls+1:number_primals-1),...
    zeros(number_inequalities,1));

% project auxiliary variable Z
primal(number_controls+number_inequalities+1) = ...
    max(primal(number_controls+number_inequalities+1),0.);

end