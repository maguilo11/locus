function [output] = maxApx(a,b,k)
output = (a*exp(k*a) + b*exp(k*b)) / (exp(k*a) + exp(k*b));
end