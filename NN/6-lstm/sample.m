hprev = zeros(1,hidden_size);
cprev = zeros(1,hidden_size);
x_now = 'a';
result = [];
for i=1:2000
    x = zeros(1,vocab_size);
    x(char_to_ix(x_now)) = 1;
    [cprev,hprev] = lstmForward(x,cprev,hprev,Wf,bf,Wi,bi,Wc,bc,Wo,bo);
    y = hprev * Why + bhy;
    p = exp(y) / sum(exp(y));
    index = discretize(rand([1,1]),[0 cumsum(p)]);
    x_now = ix_to_char(index);
    result = [result,x_now];
end
display(result);