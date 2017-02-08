addpath(genpath('..'));

images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
images = images';
images_mean = mean(images);
images = images - repmat(images_mean, size(images,1),1);

tests_images = loadMNISTImages('t10k-images-idx3-ubyte');
tests_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
tests_images = tests_images';
tests_images = tests_images - repmat(images_mean, size(tests_images,1),1);


batch_size = 128;
decay_rate = 0.001;
epoch = 10000;

W1 = random('norm',0,1,[784,120]) ./ sqrt(120);
b1 = random('norm',0,1,[1,120]) ./ sqrt(120);
W2 = random('norm',0,1,[120,100]) ./ sqrt(100);
b2 = random('norm',0,1,[1,100]) ./ sqrt(100);
W3 = random('norm',0,1,[100,10]) ./ sqrt(10);
b3 = random('norm',0,1,[1,10]) ./ sqrt(10);

for i=1:epoch
    dW1 = zeros(size(W1));
    db1 = zeros(size(b1));
    dW2 = zeros(size(W2));
    db2 = zeros(size(b2));
    dW3 = zeros(size(W3));
    db3 = zeros(size(b3));
    
    index = unidrnd(size(images,1),1,batch_size);
    batch_data = images(index,:);
    batch_label = labels(index,:);
    
    for i=1:batch_size
       h1 = batch_data(i,:)*W1 + b1;
       a1 = sigmoid(h1);
       h2 = a1 * W2 + b2;
       a2 = sigmoid(h2);
       h3 = a2 * W3 + b3;
       a3 = exp(h3);
       
       sa3 = sum(a3);
       da3 = zeros(size(a3));
       da3(:) = 1.0 / sa3;
       da3(batch_label(i)+1) = 1.0 / sa3 - 1.0 / a3(batch_label(i)+1);
       
       dh3 = da3 .* a3;
       da2 = dh3 * W3';
       tdW3 = a2' * dh3;
       tdb3 = dh3;
       
       dh2 = da2 .* dsigmoid(h2);
       da1 = dh2 * W2';
       tdW2 = a1' * dh2;
       tdb2 = dh2;
       
       dh1 = da1 .* dsigmoid(h1);
       tdW1 = batch_data(i,:)' * dh1;
       tdb1 = dh1;
       
       dW1 = dW1 + tdW1;
       db1 = db1 + tdb1;
       dW2 = dW2 + tdW2;
       db2 = db2 + tdb2;
       dW3 = dW3 + tdW3;
       db3 = db3 + tdb3;
    end
    
    W1 = W1 - dW1*decay_rate;
    W2 = W2 - dW2*decay_rate;
    W3 = W3 - dW3*decay_rate;
    b1 = b1 - db1*decay_rate;
    b2 = b2 - db2*decay_rate;
    b3 = b3 - db3*decay_rate;
    
end

n = size(images,1);
right = 0;
for i =1:n
    a1 = sigmoid(images(i,:) * W1 + b1);
    a2 = sigmoid(a1 * W2 + b2);
    o = exp(a2 * W3 + b3);
    p = o ./ sum(o);
    index = find(p == max(p));
    if index(1) == (labels(i)+1)
        right = right + 1;
    end
end

accuracy1 = right / n;

n = size(tests_images,1);
right = 0;
for i =1:n
    a1 = sigmoid(tests_images(i,:) * W1 + b1);
    a2 = sigmoid(a1 * W2 + b2);
    o = exp(a2 * W3 + b3);
    p = o ./ sum(o);
    index = find(p == max(p));
    if index(1) == (tests_labels(i)+1)
        right = right + 1;
    end
end

accuracy2 = right / n;
save W1;
save W2;
save W3;
save b1;
save b2;
save b3;
