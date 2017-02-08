function y = dReLU(x)
    y = max(x,0);
    y(find(y~=0)) = 1;
end