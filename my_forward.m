function res = my_forward(net, x)

    n = numel(net.layers) ;
    res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1)), ...
        'F', cell(1,n+1), ...
        'G', cell(1,n+1), ...
        'content_loss', num2cell(zeros(1,n+1)), ...
        'style_loss', num2cell(zeros(1,n+1))) ;
    res(1).x = x ;

    for i=1:n
        l = net.layers{i} ;
        res(i).time = tic ;
        switch l.type
            case 'conv'
                res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                    'pad', l.pad, 'stride', l.stride) ;
            case 'pool'
                res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
                    'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
            case 'relu'
                res(i+1).x = vl_nnrelu(res(i).x) ;
            case 'content_loss'
                res(i+1) = forward_content_loss(res(i), res(i+1)) ;
            case 'style_loss'
                [res(i), res(i+1)] = forward_style_loss(res(i), res(i+1)) ;
        end
        res(i).time = toc(res(i).time) ;
    end
end