fprintf('Testing gradient using forward-differencing...\n');
order = 1;
derivativeCheck(@func_to_min,mixed,order,1,net,size(content));

fprintf('Testing gradient using central-differencing...\n');
derivativeCheck(@func_to_min,mixed,order,2,net,size(content));

fprintf('Testing gradient using complex-step derivative...\n');
derivativeCheck(@func_to_min,mixed,order,3,net,size(content));

fprintf('\n\n\n');
pause

fprintf('Testing Hessian using forward-differencing\n');
order = 2;
derivativeCheck(@func_to_min,mixed,order,1,net,size(content));

fprintf('Testing Hessian using central-differencing\n');
order = 2;
derivativeCheck(@func_to_min,mixed,order,3,net,size(content));

fprintf('Testing Hessian using complex-step derivative\n');
order = 2;
derivativeCheck(@func_to_min,mixed,order,3,net,size(content));

fprintf('\n\n\n');
pause

fprintf('Testing gradient using fastDerivativeCheck...\n');
order = 1;
fastDerivativeCheck(@func_to_min,mixed,order,1,net,size(content));
fastDerivativeCheck(@func_to_min,mixed,order,2,net,size(content));
fastDerivativeCheck(@func_to_min,mixed,order,3,net,size(content));

fprintf('\n\n\n');
pause

fprintf('Testing Hessian using fastDerivativeCheck...\n');
order = 2;
fastDerivativeCheck(@func_to_min,mixed,order,1,net,size(content));
fastDerivativeCheck(@func_to_min,mixed,order,2,net,size(content));
fastDerivativeCheck(@func_to_min,mixed,order,3,net,size(content));
