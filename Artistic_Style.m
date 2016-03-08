net = load('imagenet-vgg-verydeep-19');
iteration = 1000;
init_std = 128;
learningRate = logspace(-1,-3,iteration);
alpha = 0.00000001;
beta = 0.0001;
addpath 'matlab/mex';
net.layers(22:end) = [];
net = insert_content_loss(net, 21, 1, alpha);
net = insert_style_loss(net, 21, 0.25, beta);
net = insert_style_loss(net, 12, 0.25, beta);
net = insert_style_loss(net, 7, 0.25, beta);
net = insert_style_loss(net, 2, 0.25, beta);

% net.layers(3:end) = [];
% net = insert_style_loss(net, 2, 1, beta);

% net.layers(1:end) = [];
% net.layers{1} = struct('type', 'content_loss', 'c', 1 * alpha) ;

% width = 8*32; height = 8*32;
% width = 1; height = 1;
style_image = imread('style.jpg');
max_side = 320;
if (size(style_image,1) > max_side || size(style_image,2) > max_side)
    new_h = round(min(max_side/size(style_image,1),max_side/size(style_image,2)) * size(style_image,1));
    new_w = round(min(max_side/size(style_image,2),max_side/size(style_image,1)) * size(style_image,2));
    style = single(imresize(style_image,[new_h,new_w]));
else
    style = single(style_image);
end
style = bsxfun(@minus,style,net.meta.normalization.averageImage);
style_res = my_forward(net, style);

% content_image = imread('content.jpg');
content_image = imread('content.jpg');
if (size(content_image,1) > max_side || size(content_image,1) > max_side)
    new_h = round(min(max_side/size(content_image,1),max_side/size(content_image,2)) * size(content_image,1));
    new_w = round(min(max_side/size(content_image,2),max_side/size(content_image,1)) * size(content_image,2));
    content = single(imresize(content_image,[new_h,new_w]));
else
    content = single(content_image);
end
content = bsxfun(@minus,content,net.meta.normalization.averageImage);
content_res = my_forward(net, content);

for i = 1:numel(net.layers)
    switch net.layers{i}.type
        case 'content_loss'
            net.layers{i}.content = content_res(i).x;
        case 'style_loss'
            net.layers{i}.style = style_res(i).G;
    end
end

% mixed = init_std * rand(size(content),'single');
mixed = content;

mixed = reshape(mixed,numel(mixed),1);
options.Method = 'lbfgs';
options.useMex = 0;
% options.display = 'none';
mixed = minFunc(@func_to_min,mixed,options,net,size(content));
mixed_image = bsxfun(@plus,reshape(mixed,size(content)),net.meta.normalization.averageImage);
imshow(mixed_image/255);

% for n = 1:iteration
%     [f,g] = func_to_min(mixed,net,size(content));
%     mean_g = mean(mean(mean(abs(g))));
%     mixed = mixed - learningRate(n) / mean_g * g;
%     mixed_image = bsxfun(@plus,reshape(mixed,size(content)),net.meta.normalization.averageImage);
%     
%     fprintf('iteration: %d; mean_dzdx: %f;\n',n,mean_g);
%     imshow(mixed_image/255);
%     drawnow;
% end

% L_style = w_l * Es';

