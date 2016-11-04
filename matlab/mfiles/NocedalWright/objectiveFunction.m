function [Operators] = objectiveFunction()
Operators.evaluate=@(primal) evaluate(primal);
Operators.firstDerivative=@(primal) firstDerivative(primal);
Operators.secondDerivative=@(primal,dprimal) secondDerivative(primal, dprimal);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluate(primal)
    % J(X) = exp(x1 * x2 * x3 * x4 * x5) - 0.5 * (1 + x1^3 + x2^3)^2
    a = exp(primal(1)*primal(2)*primal(3)*primal(4)*primal(5));
    b = 1. + power(primal(1), 3.) + power(primal(2), 3.);
    c = 0.5 * power(b, 2.);
    output = a - c;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivative(primal)
    % (1 + x1^3 + x2^3)
    a = 1. + power(primal(1), 3) + power(primal(2), 3.);
    % exp(x1 * x2 * x3 * x4 * x5)
    b = exp(primal(1) * primal(2) * primal(3) * primal(4) * primal(5));
    % J_x1(X) = -3. * x1^2 * (1 + x1^3 + x2^3) + exp(x1 * x2 * x3 * x4 * x5) * x2 * x3 * x4 * x5
    output(1) = (-3. * power(primal(1), 2) * a) ...
        + (b * primal(2) * primal(3) * primal(4) * primal(5));
    % J_x2(X) = -3. * x2^2 * (1 + x1^3 + x2^3) + exp(x1 * x2 * x3 * x4 * x5) * x1 * x3 * x4 * x5
    output(2) = (-3. * power(primal(2), 2) * a) ...
        + (b * primal(1) * primal(3) * primal(4) * primal(5));
    % exp(x1 * x2 * x3 * x4 * x5) * x1 * x2 * x4 * x5
    output(3) = b * primal(1) * primal(2) * primal(4) * primal(5);
    % exp(x1 * x2 * x3 * x4 * x5) * x1 * x2 * x3 * x5
    output(4) = b * primal(1) * primal(2) * primal(3) * primal(5);
    % exp(x1 * x2 * x3 * x4 * x5) * x1 * x2 * x3 * x4
    output(5) = b * primal(1) * primal(2) * primal(3) * primal(4);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = secondDerivative(primal,dprimal)
    x1 = primal(1);
    x2 = primal(2);
    x3 = primal(3);
    x4 = primal(4);
    x5 = primal(5);
    e = exp(x1*x2*x3*x4*x5);
    % column 1
    % exp(x1 x2 x3 x4 x5) x2^2 x3^2 x4^2 x5^2 - 6 x1 (x1^3 + x2^3 + 1) - 9 x1^4
    H11 = e * power(x2, 2.) * power(x3, 2.) * power(x4, 2.) * power(x5, 2.) ...
            - 6. * x1 * (power(x1, 3.) + power(x2, 3.) + 1.) ...
            - 9. * power(x1, 4.);
    % -9. x1^2 x2^2 + exp(x1 x2 x3 x4 x5) x3 x4 x5 + exp(x1 x2 x3 x4 x5) x1 x2 x3^2 x4^2 x5^2
    H12 = e * x1 * x2 * power(x3, 2.) * power(x4, 2.) * power(x5, 2.) ...
            + x3 * x4 * x5 * e ...
            - 9. * power(x1, 2.) * power(x2, 2.);
    % E^(x1 x2 x3 x4 x5) x2 x4 x5 + E^(x1 x2 x3 x4 x5) x1 x2^2 x3 x4^2 x5^2
    H13 = ( e * x2 * x4 * x5 ) ...
            + ( e * x1 * power(x2, 2.) * x3 * power(x4, 2.) * power(x5, 2.) );
    % E^(x1 x2 x3 x4 x5) x2 x3 x5 + E^(x1 x2 x3 x4 x5) x1 x2^2 x3^2 x4 x5^2
    H14 = ( e * x2 * x3 * x5 ) ...
            + ( e * x1 * power(x2, 2.) * power(x3, 2) * x4 * power(x5, 2.) );
    % E^(x1 x2 x3 x4 x5) x2 x3 x4 + E^(x1 x2 x3 x4 x5) x1 x2^2 x3^2 x4^2 x5
    H15 = ( e * x2 * x3 * x4 ) ...
            + ( e * x1 * power(x2, 2.) * power(x3, 2.) * power(x4, 2.) * x5 );
    % column 2
    % -9. x2^4 - 6. x2 (1 + x1^3 + x2^3) + E^(x1 x2 x3 x4 x5) x1^2 x3^2 x4^2 x5^2
    H22 = ( e * power(x1, 2.) * power(x3, 2.) * power(x4, 2.) * power(x5, 2.) ) ...
            - ( 9. * power(x2, 4.) ) ...
            - ( 6. * x2 * (1. + power(x1, 3.) + power(x2, 3.)) );
    % E^(x1 x2 x3 x4 x5) x1 x4 x5 + E^(x1 x2 x3 x4 x5) x1^2 x2 x3 x4^2 x5^2
    H23 = ( e * x1 * x4 * x5 ) ...
            + ( e * power(x1, 2.) * x2 * x3 * power(x4, 2.) * power(x5, 2.) );
    % E^(x1 x2 x3 x4 x5) x1 x3 x5 + E^(x1 x2 x3 x4 x5) x1^2 x2 x3^2 x4 x5^2
    H24 = ( e * x1 * x3 * x5 ) ...
            + ( e * power(x1, 2.) * x2 * power(x3, 2.) * x4 * power(x5, 2.) );
    % E^(x1 x2 x3 x4 x5) x1 x3 x4 + E^(x1 x2 x3 x4 x5) x1^2 x2 x3^2 x4^2 x5
    H25 = ( e * x1 * x3 * x4 ) ...
            + ( e * power(x1, 2.) * x2 * power(x3, 2.) * power(x4, 2.) * x5 );
    % column 3
    % E^(x1 x2 x3 x4 x5) x1^2 x2^2 x4^2 x5^2
    H33 = e * power(x1, 2.) * power(x2, 2.) * power(x4, 2.) * power(x5, 2.);
    % E^(x1 x2 x3 x4 x5) x1 x2 x5 + E^(x1 x2 x3 x4 x5) x1^2 x2^2 x3 x4 x5^2
    H34 = ( e * x1 * x2 * x5 ) ...
            + ( e * power(x1, 2.) * power(x2, 2.) * x3 * x4 * power(x5, 2.) );
    % E^(x1 x2 x3 x4 x5) x1 x2 x4 + E^(x1 x2 x3 x4 x5) x1^2 x2^2 x3 x4^2 x5
    H35 = ( e * x1 * x2 * x4 ) ...
            + ( e * power(x1, 2.) * power(x2, 2.) * x3 * power(x4, 2.) * x5 );
    % column 4
    % E^(x1 x2 x3 x4 x5) x1^2 x2^2 x3^2 x5^2
    H44 = e * power(x1, 2.) * power(x2, 2.) * power(x3, 2.) * power(x5, 2.);
    % E^(x1 x2 x3 x4 x5) x1 x2 x3 + E^(x1 x2 x3 x4 x5) x1^2 x2^2 x3^2 x4 x5
    H45 = ( e * x1 * x2 * x3 ) ...
            + ( e * power(x1, 2.) * power(x2, 2.) * power(x3, 2.) * x4 * x5 );
    % column 5
    % E^(x1 x2 x3 x4 x5) x1^2 x2^2 x3^2 x4^2
    H55 = e * power(x1, 2.) * power(x2, 2.) * power(x3, 2.) * power(x4, 2.);
    % Hessian-vector product
    output(1) = H11 * dprimal(1) + H12 * dprimal(2) ...
        + H13 * dprimal(3) + H14 * dprimal(4) + H15 * dprimal(5);
    output(2) = H12 * dprimal(1) + H22 * dprimal(2) ...
        + H23 * dprimal(3) + H24 * dprimal(4) + H25 * dprimal(5);
    output(3) = H13 * dprimal(1) + H23 * dprimal(2) ...
        + H33 * dprimal(3) + H34 * dprimal(4) + H35 * dprimal(5);
    output(4) = H14 * dprimal(1) + H24 * dprimal(2) ...
        + H34 * dprimal(3) + H44 * dprimal(4) + H45 * dprimal(5);
    output(5) = H15 * dprimal(1) + H25 * dprimal(2) ...
        + H35 * dprimal(3) + H45 * dprimal(4) + H55 * dprimal(5);
end