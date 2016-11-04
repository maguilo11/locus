function [Operators] = inequalityConstraint()
Operators.value=@(index) value(index);
Operators.evaluate=@(primal,index) evaluate(primal);
Operators.firstDerivative=@(primal,index) firstDerivative(primal,index);
Operators.secondDerivative=@(primal,dprimal,index) secondDerivative(primal,dprimal,index);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = value(index)
output = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluate(primal,index)
term_one = 61. / power(primal(1), 3.);
term_two = 37. / power(primal(2), 3.);
term_three = 19. / power(primal(3), 3.);
term_four = 7. / power(primal(4), 3.);
term_five = 1. / power(primal(5), 3.);
output = term_one + term_two + term_three + term_four + term_five;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivative(primal,index)
factor = -3.;
output(1) = factor * 61. / power(primal(1), 4.);
output(2) = factor * 37. / power(primal(2), 4.);
output(3) = factor * 19. / power(primal(3), 4.);
output(4) = factor * 7. / power(primal(4), 4.);
output(5) = factor * 1. / power(primal(5), 4.);
end
