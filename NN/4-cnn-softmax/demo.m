addpath(genpath('..'));

% for convolution layer, input and output always be 3-dim tensor.
% input output:[D,R,C]
% and kernel always be 4-dim tensor.
% kernel: [nD,d,r,c]

images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
images = images';
images_mean = mean(images);
images = images - repmat(images_mean, size(images,1),1);
timages = zeros(size(images,1),28,28); %images 60000*28*28
timages(:) = images(:);
images = timages;

tests_images = loadMNISTImages('t10k-images-idx3-ubyte');
tests_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
tests_images = tests_images';
tests_images = tests_images - repmat(images_mean, size(tests_images,1),1);
ttests_images = zeros(size(tests_images,1),28,28);
ttests_images(:) = tests_images(:);
tests_images = ttests_images;


batch_size = 128;
decay_rate = 0.001;
epoch = 20000;
mu = 0.9;

% initlize is very important ,and this init are not xivar.but this work
W1 = random('norm',0,1,[10,1,3,3]) ./ sqrt(10*3*3);%cnn output 10 * 26 * 26
b1 = random('norm',0,1,[10,1]);% 
W2 = random('norm',0,1,[5,10,5,5]) ./ sqrt(5*10*5*5);%cnn output 5 * 22 * 22
b2 = random('norm',0,1,[5,1]) ./ sqrt(5);
W3 = random('norm',0,1,[1,5,5,5]) ./ sqrt(5*5*5); %cnn output 1 * 17 * 17
b3 = random('norm',0,1,[1]);
W4 = random('norm',0,1,[10,18*18]) ./ 18;
b4 = random('norm',0,1,[10,1]) ./ sqrt(10);
    %index = unidrnd(size(images,1),1,batch_size);
for i=1:epoch
    display('epoch:')
    display(i);
    dW1 = zeros(size(W1));
    VW1 = zeros(size(W1));
    db1 = zeros(size(b1));
    Vb1 = zeros(size(b1));
    dW2 = zeros(size(W2));
    VW2 = zeros(size(W2));
    db2 = zeros(size(b2));
    Vb2 = zeros(size(b2));
    dW3 = zeros(size(W3));
    VW3 = zeros(size(W3));
    db3 = zeros(size(b3));
    Vb3 = zeros(size(b3));
    dW4 = zeros(size(W4));
    VW4 = zeros(size(W4));
    db4 = zeros(size(b4));
    Vb4 = zeros(size(b4));
    
    index = unidrnd(size(images,1),1,batch_size);
    batch_data = images(index,:,:);
    batch_label = labels(index,:,:);
    loss = 0;
    for i=1:batch_size
		input1 = batch_data(i,:,:);
		output1 = ConvForward(input1, W1, b1);
		Routput1 = ReLU(output1);
		output2 = ConvForward(Routput1, W2, b2);
		Routput2 = ReLU(output2);
		output3 = ConvForward(Routput2, W3, b3);
		Routput3 = ReLU(output3);
		output4 = W4 * Routput3(:) + b4;
		a = exp(output4);
		
        sa = sum(a);
        loss = loss + (-log(a(batch_label(i)+1) / sa));
        da = zeros(size(a));
        da(:) = 1.0 / sa;
        da(batch_label(i)+1) = 1.0 / sa - 1.0 / a(batch_label(i) + 1);
        
        doutput4 = zeros(size(output4));
        doutput4(:) = da .* a;
        tdW4 = doutput4 * Routput3(:)';
        tdb4 = doutput4;
        dRoutput3 = zeros(size(Routput3));
        dRoutput3(:) = W4' * doutput4;
        
        doutput3 = dRoutput3 .* dReLU(output3);
        
        [dRoutput2,tdW3,tdb3] = ConvBackward(doutput3, output2, W3, b3);
        
        doutput2 = dRoutput2 .* dReLU(output2);
        
        [dRoutput1,tdW2,tdb2] = ConvBackward(doutput2, output1, W2, b2);
        
        doutput1 = dRoutput1 .* dReLU(output1);
        
        [dinput1,tdW1,tdb1] = ConvBackward(doutput1, input1, W1, b1);
                        
       dW1 = dW1 + tdW1;
       db1 = db1 + tdb1;
       dW2 = dW2 + tdW2;
       db2 = db2 + tdb2;
       dW3 = dW3 + tdW3;
       db3 = db3 + tdb3;
		dW4 = dW4 + tdW4;
		db4 = db4 + tdb4;
    end
    display(loss);
    V_prev = VW1;
    VW1 = mu * VW1 - tdW1 * decay_rate;
    W1 = W1 - mu * V_prev + (1 + mu)* VW1;
    
    V_prev = VW2;
    VW2 = mu * VW2 - tdW2 * decay_rate;
    W2 = W2 - mu * V_prev + (1 + mu)* VW2;
    
    V_prev = VW3;
    VW3 = mu * VW3 - tdW3 * decay_rate;
    W3 = W3 - mu * V_prev + (1 + mu)* VW3;
    
    V_prev = VW4;
    VW4 = mu * VW4 - tdW4 * decay_rate;
    W4 = W4 - mu * V_prev + (1 + mu)* VW4;
    
    b1 = b1 - db1*decay_rate;
    b2 = b2 - db2*decay_rate;
    b3 = b3 - db3*decay_rate;
	b4 = b4 - db4*decay_rate;
    
end
% 
% n = size(images,1);
% right = 0;
% for i =1:n
% 		input1 = images(i,:,:);
% 		output1 = ConvForward(input1, W1, b1);
% 		Routput1 = ReLU(output1);
% 		output2 = ConvForward(Routput1, W2, b2);
% 		Routput2 = ReLU(output2);
% 		output3 = ConvForward(Routput2, W3, b3);
% 		Routput3 = ReLU(output3);
% 		output4 = W4 * Routput3(:) + b4;
% 		a = exp(output4);
%     p = a ./ sum(a);
%     index = find(p == max(p));
%     if index(1) == (labels(i)+1)
%         right = right + 1;
%     end
% end
% 
% accuracy1 = right / n;

n = size(tests_images,1);
right = 0;
n = 10;
for i =1:n
    display('test:');
    display(i);
		input1 = tests_images(i,:,:);
		output1 = ConvForward(input1, W1, b1);
		Routput1 = ReLU(output1);
		output2 = ConvForward(Routput1, W2, b2);
		Routput2 = ReLU(output2);
		output3 = ConvForward(Routput2, W3, b3);
		Routput3 = ReLU(output3);
		output4 = W4 * Routput3(:) + b4;
		a = exp(output4);
    p = a ./ sum(a);
    index = find(p == max(p));
    if index(1) == (tests_labels(i)+1)
        right = right + 1;
    end
end

accuracy2 = right / n;
