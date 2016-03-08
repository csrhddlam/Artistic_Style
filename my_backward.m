function res = my_backward(net, res)

    n = numel(net.layers) ;
    for i=n:-1:1
        l = net.layers{i} ;
        res(i).backwardTime = tic ;
        switch l.type
            case 'conv'
                [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                    vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                    res(i+1).dzdx, ...
                    'pad', l.pad, 'stride', l.stride);
            case 'pool'
                res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                    'pad', l.pad, 'stride', l.stride, ...
                    'method', l.method);
            case 'relu'
                res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
            case 'content_loss'
                if (i == n)
                    res(i) = backward_content_loss(l,res(i),zeros(size(res(i).x)));
                else
                    res(i) = backward_content_loss(l,res(i),res(i+1).dzdx);
                end
            case 'style_loss'
                if (i == n)
                    res(i) = backward_style_loss(l,res(i),zeros(size(res(i).x)));
                else
                    res(i) = backward_style_loss(l,res(i),res(i+1).dzdx);
                end

        end
        res(i).backwardTime = toc(res(i).backwardTime) ;
    end
end