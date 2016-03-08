function input = backward_content_loss(l,input,dzdx)
% input.dzdx = (input.x - l.content) / numel(l.content) + dzdx;
% input.content_loss = 0.5 * immse(input.x,l.content);
input.dzdx = l.c * (input.x - l.content) + dzdx;
% input.content_loss = 0.5 * l.c * immse(input.x,l.content) * numel(l.content);
input.content_loss = 0.5 * l.c * sum(sum(sum((input.x-l.content).* (input.x-l.content))));
end