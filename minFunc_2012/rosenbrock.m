function [f,g] = rosenbrock(x)
f = sum((x-1) .* (x-1)) ;
g = 10 .* (x-1);