function [dW, p] = runNet( W, batch_data, batch_label, model )
dW={};
dW.W1 = zeros(size(W.W1));
dW.b1 = zeros(size(W.b1));
dW.W2 = zeros(size(W.W2));
dW.b2 = zeros(size(W.b2));
dW.W3 = zeros(size(W.W3));
dW.b3 = zeros(size(W.b3));
n = size(batch_data,1);

for i=1:n
   h1 = batch_data(i,:)*W.W1 + W.b1;
   a1 = max(h1, 0); % ReLU
   h2 = a1*W.W2 + W.b2;
   a2 = max(h2, 0); % ReLU
   o = a2*W.W3 + W.b3;
   p = softmax(o);
   if model == 'train'
     label_index = batch_label(i)+1;
     do = zeros(size(o));
       do = p;
      do(label_index) = p(label_index) - 1;
      dW3 = a2'*do;
       db3 = do;
       da2 = do*W.W3';
       dh2 = zeros(size(h2));
       dh2_index = find(h2>0);
      dh2(dh2_index) = da2(dh2_index);
      dW2 = a1'*dh2;
      db2 = dh2;
       da1 = dh2*dW2';
       dh1 = zeros(size(h1));
       dh1_index = find(h1>0);
      dh1(dh1_index) = da1(dh1_index);
       dW1 = batch_data(i,:)'*dh1;
       db1 = dh1;
       dW.W1 = dW.W1 +dW1 ./ n;
       dW.b1 = dW.b1 + db1 ./ n;
       dW.W2 = dW.W2 +dW2 ./ n;
       dW.b2 = dW.b2 + db2 ./ n;
       dW.W3 = dW.W3 + dW3 ./ n;
       dW.b3 = dW.b3+db3 ./ n;
   end
end

    function p = softmax(o)
        p = exp(o);
        s = sum(p);
        p = p / s;
    end
end

