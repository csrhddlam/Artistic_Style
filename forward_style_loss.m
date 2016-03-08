function [input,output] = forward_style_loss(input,output)
output.x = input.x;
M = size(input.x,1) * size(input.x,2);
N = size(input.x,3);
input.F = reshape(input.x,M,N);
input.G = input.F' * input.F / M;
end