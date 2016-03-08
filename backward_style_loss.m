function input = backward_style_loss(l,input,dzdx)
M = size(input.x,1) * size(input.x,2);
N = size(input.x,3);
input.style_loss = 0.25 * l.w * immse(input.G,l.style);
dEdF = input.F * (input.G - l.style) / M / N / N;
input.dzdx = l.w * reshape(dEdF,size(input.x,1),size(input.x,2),size(input.x,3)) + dzdx;
end