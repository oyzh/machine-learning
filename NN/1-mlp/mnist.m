addpath(genpath('.'));
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

images = images';
images_mean = mean(images);
images = images - repmat(images_mean, size(images,1),1);
sigma = std(images,0,1).^2;
sigma(find(sigma == 0)) = 1;
%images = images ./ repmat(sigma, size(images,1),1);
% image = images(1,:);
% I = zeros(28,28);
% I(:) = image';
% 
% imshow(I);
batch_size = 128;       
decay_rate = 0.01;
epoch = 1000;

X_l = 28*28;
W_l = [100,100];
b_l = W_l;

W = {};
dW = {};
W.W1 = random('norm',0,1,[784,100]) ./ sqrt(100);
W.b1 = random('norm',0,1,[1,100]) ./ sqrt(100);
W.W2 = random('norm',0,1,[100,100]) ./ sqrt(100);
W.b2 = random('norm',0,1,[1,100]) ./ sqrt(100);
W.W3 = random('norm',0,1,[100,10]) ./ sqrt(10);
W.b3 = random('norm',0,1,[1,10]) ./ sqrt(10);



for i=1:10000
    index = unidrnd(size(images,1),1,batch_size);
    batch_data = images(index,:);
    batch_label = labels(index,:);
    [dW,p] = runNet(W, batch_data, batch_label,'train');
    
    W.W1 = W.W1 - dW.W1*decay_rate;
    W.W2 = W.W2 - dW.W2*decay_rate;
    W.W3 = W.W3 - dW.W3*decay_rate;
    W.b1 = W.b1 - dW.b1*decay_rate;
    W.b2 = W.b2 - dW.b2*decay_rate;
    W.b3 = W.b3 - dW.b3*decay_rate;
    
end

[dW, p] = runNet(W, images(1,:),labels(1),'testt');






