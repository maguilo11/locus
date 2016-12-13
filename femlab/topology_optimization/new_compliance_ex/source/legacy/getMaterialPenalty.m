function [quantity] = getMaterialPenalty(control, model_t, derivative_t)

switch model_t
    case 'simp'
        quantity = getSimpModelPenalty(control, derivative_t);
    case 'ramp'
        quantity = getSimpModelPenalty(control, derivative_t);
end

end