function net = insert_content_loss(net, l, c, alpha)
% --------------------------------------------------------------------
layer = struct('type', 'content_loss', 'c', c * alpha) ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;