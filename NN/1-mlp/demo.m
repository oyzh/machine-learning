addpath(genpath('.'));
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

images = images';
images_mean = mean(images);
images = images - repmat(images_mean, size(images,1),1);

batch_size = 128;
decay_rate = 0.001;

W1 = random('norm',0,1,[784,100]) ./ sqrt(100);
b1 = random('norm',0,1,[1,100]) ./ sqrt(100);
W2 = random('norm',0,1,[100,1]) ./ sqrt(10);
b2 = random('norm',0,1,[1,1]) ./ sqrt(10);

% for i=1:100000
%     dW1 = zeros(size(W1));
%     db1 = zeros(size(b1));
%     dW2 = zeros(size(W2));
%     db2 = zeros(size(b2));
%     
%     index = unidrnd(size(images,1),1,batch_size);
%     batch_data = images(index,:);
%     batch_label = labels(index,:);
%     
%     for i=1:batch_size
%        h1 = batch_data(i,:)*W1 + b1;
%        a1 = sigmoid(h1);
%        o = a1 * W2 + b2;
%        do = o - batch_label(i);
%        da1 = (do * W2)';
%        tdW2 = (do * a1)';
%        tdb2 = do;
%        dh1 = da1 .* dsigmoid(h1);
%        tdW1 = batch_data(i,:)'*dh1;
%        tdb1 = dh1;
%        
%        dW1 = dW1 + tdW1;
%        db1 = db1 + tdb1;
%        dW2 = dW2 + tdW2;
%        db2 = db2 + tdb2;
%        
%     end
%     
%     dW1 = dW1 ./ batch_size;
%     db1 = db1 ./ batch_size;
%     dW2 = dW2 ./ batch_size;
%     db2 = db2 ./ batch_size;
%     
%     W1 = W1 - dW1*decay_rate;
%     W2 = W2 - dW2*decay_rate;
%     b1 = b1 - db1*decay_rate;
%     b2 = b2 - db2*decay_rate;
%     
% end

load W1
load W2 
load b1
load b2
n = size(images,1);
right = 0;
for i =1:n
    image = images(i,:);
    label = labels(i);
    h1 = image*W1 + b1;
    a1 = sigmoid(h1);
    o = a1 * W2 + b2;
    if round(o) == label
        right = right + 1;
    end
end

accuracy = right / n;
