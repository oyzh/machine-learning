function lossvalue = Loss(a, label)
    p =  a / sum(a);
    lossvalue = -log(p(label+1));
end