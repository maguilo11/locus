function [Operators] = equalityConstraint()
Operators.residual=@(primal) residual(primal);
Operators.firstDerivative=@(primal,dprimal) firstDerivative(primal,dprimal);
Operators.adjointFirstDerivative=@(primal,dual) adjointFirstDerivative(primal,dual);
Operators.secondDerivative=@(primal,dual,dprimal) secondDerivative(primal,dual,dprimal);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = residual(primal)
    % C1 = x1^2 + x2^2 + x3^2 + x4^2 + x5^2 - 10
    output(1) = power(primal(1), 2.) + power(primal(2), 2.) + ...
        power(primal(3), 2.) + power(primal(4), 2.) + ...
        power(primal(5), 2.) - 10;
    % C2 = x2 * x3 - 5. * x4 * x5
    output(2) = primal(2) * primal(3) - (5. * primal(4) * primal(5));
    % C3 = x1^3 + x2^3 + 1.
    output(3) = power(primal(1), 3.) + power(primal(2), 3.) + 1.;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivative(primal,dprimal)
    % |  2*x1    2*x2   2*x3   2*x4   2*x5 |
    % |   0       x3     x2   -5*x5  -5*x4 |
    % | 3*x1^2  3*x2^2   0      0      0   |
    output(1) = 2. * primal(1) * dprimal(1) + 2. * primal(2) * dprimal(2) ...
        + 2. * primal(3) * dprimal(3) + 2. * primal(4) * dprimal(4) ...
        + 2. * primal(5) * dprimal(5);
    output(2) = primal(3) * dprimal(2) + primal(2) * dprimal(3) ...
        - 5. * primal(5) * dprimal(4) - 5. * primal(4) * dprimal(5);
    output(3) = 3. * power(primal(1), 2.) * dprimal(1) ...
        + 3 * power(primal(2), 2.) * dprimal(2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = adjointFirstDerivative(primal,dual)
    % | 2*x1    0    3*x1^2 |
    % | 2*x2    x3   3*x2^2 |
    % | 2*x3    x2     0    |
    % | 2*x4  -5*x5    0    |
    % | 2*x5  -5*x4    0    |
    output(1) = 2. * primal(1) * dual(1) ...
        + 3. * power(primal(1), 2.) * dual(3);
    output(2) = 2. * primal(2) * dual(1) + primal(3) * dual(2) ...
        + 3. * power(primal(2), 2.) * dual(3);
    output(3) = 2. * primal(3) * dual(1) + primal(2) * dual(2);
    output(4) = 2. * primal(4) * dual(1) - 5. * primal(5) * dual(2);
    output(5) = 2. * primal(5) * dual(1) - 5. * primal(4) * dual(2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivative(primal,dual,dprimal)
    output(1) = 2. * dual(1) * dprimal(1) + ...
        6. * primal(1) * dual(3) * dprimal(1);
    output(2) = 2. * dual(1) * dprimal(2) + ...
        6. * primal(2) * dual(3) * dprimal(2) + dual(2) * dprimal(3);
    output(3) = dual(2) * dprimal(2) + 2. * dual(1) * dprimal(3);
    output(4) = 2. * dual(1) * dprimal(4) - 5. * dual(2) * dprimal(5);
    output(5) = -5. * dual(2) * dprimal(4) + 2. * dual(1) * dprimal(5);
end