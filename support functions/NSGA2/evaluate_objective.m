function f = evaluate_objective(x, M, V)
%
% function f = evaluate_objective(x, M, V)
%
% Function to evaluate the objective functions for the given input vector x.%
% x is an array of decision variables and f(1), f(2), etc are the
% objective functions. The algorithm always minimizes the objective
% function hence if you would like to maximize the function then multiply
% the function by negative one. M is the numebr of objective functions and
% V is the number of decision variables. 
%
% This functions is basically written by the user who defines his/her own
% objective function. Make sure that the M and V matches your initial user
% input.
%

x = x(1:V) ;
x = x(:)   ;

% --------------------------------------
% insert here your function:
global opt_inputs;
n = opt_inputs.n;
h_init = opt_inputs.h_init;
param = opt_inputs.param;


%policy 
param.reg.h1 = x(1);
param.reg.h2 = x(2);
param.reg.m1 = x(3);
param.reg.m2 = x(4);

%run lake simulation
[s_reg, h_reg, r_reg] = simulate_dam(n, h_init, param, 'reg');

%compute objectives
h_reg = h_reg(2:end);
s_reg = s_reg(2:end);
r_reg = r_reg(2:end);

%irrigation deficit 
 w= param.reg.w;
 def = max(w-r_reg, 0);
 Jir_reg = mean( def.^2);
 
 
%flood 
%S_flo_reg = zeros(size(h_reg));
idx = (h_reg > opt_inputs.h_flo);
%S_flo_reg(idx) = 0.081*h_reg(idx).^3 - 0.483*h_reg(idx).^2 + 1.506*h_reg(idx) -1.578;
Jflo_reg = sum(idx)/opt_inputs.Ny;

f = [Jir_reg, Jflo_reg];

% --------------------------------------

% Check for error
if length(f) ~= M
    error('The number of decision variables does not match you previous input. Kindly check your objective function');
end