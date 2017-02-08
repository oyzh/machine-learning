function y = sigmoid(x)
    x=double(x);
    y = 1.0 ./ (1+exp(-x));
end