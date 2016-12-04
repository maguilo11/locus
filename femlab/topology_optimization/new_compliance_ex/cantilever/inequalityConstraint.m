function [Operators] = inequalityConstraint()
Operators.value=@(index)value(index);
Operators.evaluate=@(state,control,index)evaluate(state,control,index);
Operators.gradient=@(state,control,index)gradient(state,control,index);
end
% Evaluate inequality constraint residual
function [output] = value(state,control,index)
output = 1;
end
% Evaluate inequality constraint
function [output] = evaluate(state,control,index)

term_one = (61.) / power(control(1), 3.);
term_two = (37.) / power(control(2), 3.);
term_three = (19.) / power(control(3), 3.);
term_four = (7.) / power(control(4), 3.);
term_five = (1.) / power(control(5), 3.);

output = term_one + term_two + term_three + term_four + term_five;

end
% Compute inequality constraint gradient
function [output] = gradient(state,control,index)

factor = -3.;

output(1) = factor * ((61.) /power(control(1), 4.));
output(2) = factor * ((37.) /power(control(2), 4.));
output(3) = factor * ((19.) /power(control(3), 4.));
output(4) = factor * ((7.) /power(control(4), 4.));
output(5) = factor * ((1.) /power(control(5), 4.));

end