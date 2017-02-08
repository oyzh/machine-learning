function output = ConvForward(input, W, b)
% input and output are 3-dim
% W is 4 dim
% b is 1 dim vector
    [D,R,C] = size(input);
    [nD,D,r,c] = size(W); % we should make sure that input's D equal to W's D
    img_size = R; % we just make R equal to C
    output = zeros([nD, img_size-r+1, img_size-c+1]); % we use the 'valid' conv
    for i = 1:nD
        k = W(i,:,:,:);
        k = reshape(k,size(k,2),size(k,3),size(k,4));
        output(i,:,:) = ThreeConv(input,k) + b(i);
    end
    
    function re = ThreeConv(data, kernel)
        n = size(data,1); % size(kernel,1) should same
        re = zeros(size(data,2)-size(kernel,2)+1,size(data,3)-size(kernel,3)+1);
        dim2data = zeros(size(data,2),size(data,3));
        dim2ker = zeros(size(kernel,2),size(kernel,3));
        for j = 1:n
            dim2data(:) = data(j,:,:);
            dim2ker(:) = kernel(j,:,:);
            conv_r = filter2(dim2ker,dim2data,'valid');
            re = re + conv_r;
        end
    end
end
