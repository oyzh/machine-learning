function [dinput,dW,db] = ConvBackward(doutput,input, W, b)
% input and output are 3-dim
% W is 4 dim
% b is 1 dim vector
	db = zeros(size(b));
    db(:) = sum(sum(doutput,2),3);
    dinput = zeros(size(input));
    dW = zeros(size(W));
    
    n = size(dinput,1);
    
    k = zeros(size(W,1),size(W,3),size(W,4)); % dinput can use the sym kernel conv
    for i=1:n
        k(:) = W(:,i,:,:);
       dinput(i,:,:) = dconv( doutput, k);
    end
    
    n = size(W,1);
    dataout = zeros(size(doutput,2),size(doutput,3));
    for i=1:n
        dataout(:) = doutput(i,:,:);
       dW(i,:,:,:) = dconv2(input, dataout); 
    end
    
    
    function dy = dconv(do, kernel)
        dy = zeros(size(do,2)-1+size(kernel,2), size(do,3)-1+size(kernel,3));
        len = size(do,1);
        data = zeros(size(do,2),size(do,3));
        ke = zeros(size(kernel,2),size(kernel,3));
        for j=1:len
            data(:) = do(j,:,:);
            ke(:) = kernel(j,:,:);
            %ke = flipdim(flipdim(ke,2),1); % two sym IMPORTANT
           %dy = dy + conv2(data,ke);
           dy = dy + conv2(data,ke);
        end
    end
    function dy = dconv2(in, dout)
        dy = zeros(size(in,1),size(in,2)+1-size(dout,1),size(in,3)+1-size(dout,2));
        n1 = size(in,1);
        d1 = zeros(size(in,2),size(in,3));
        for j =1:n1
            d1(:) = in(j,:,:);
           dy(j,:,:) = filter2( dout, d1,'valid');
           %dy(j,:,:) = conv2(d1,dout,'valid');
        end
    end
end
