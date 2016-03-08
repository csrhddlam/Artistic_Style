function net = insert_style_loss(net, l, w, beta)
% --------------------------------------------------------------------
layer = struct('type', 'style_loss', 'w', w * beta) ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;