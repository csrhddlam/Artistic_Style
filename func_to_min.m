function [f,g] = func_to_min(mixed,net,size)
    mixed = reshape(mixed,size);
    mixed_res = my_forward(net, mixed);
    mixed_res = my_backward(net, mixed_res);
    mixed_image = bsxfun(@plus,mixed,net.meta.normalization.averageImage);
    L_content = 0; L_style = 0;
    for i = 1:numel(net.layers)
        switch net.layers{i}.type
            case 'content_loss'
                L_content = L_content + mixed_res(i).content_loss;
            case 'style_loss'
                L_style = L_style + mixed_res(i).style_loss;
        end
    end
    f = L_content + L_style;
    g = reshape(mixed_res(1).dzdx,numel(mixed),1);
    fprintf('content_loss: %f; style_loss: %f;\n',L_content,L_style);
%     fprintf('iteration: %d; mean_dzdx: %f; content_loss: %f; style_loss: %f;\n',n,mean_dzdx,L_content,L_style);
    imshow(mixed_image/255);
    drawnow;
end