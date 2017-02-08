
seq_length = 25;
hprev = zeros(1,hidden_size);
cprev = zeros(1,hidden_size);
p = 1;
inputs = {}; targets = {}; xs = {}; cs = {}; hs ={}; ys = {};ps = {};
first_hprev = hprev;
first_cprev = cprev;
loss = 0;

for i = p:(p+seq_length - 1)
    inputs{i} = char_to_ix(data(i));
    targets{i} = char_to_ix(data(i+1));
    xs{i} = zeros(1,vocab_size);
    xs{i}(inputs{i}) = 1;
    [cprev,hprev] = lstmForward(xs{i},cprev,hprev,Wf,bf,Wi,bi,Wc,bc,Wo,bo);
    cs{i} = cprev;
    hs{i} = hprev;
    ys{i} = hprev * Why + bhy;
    ps{i} = exp(ys{i}) / sum(exp(ys{i}));
    loss = loss - log(ps{i}(targets{i}));
end

dWf = zeros(size(Wf));dbf = zeros(size(bf));
dWi = zeros(size(Wi));dbi = zeros(size(bi));
dWc = zeros(size(Wc));dbc = zeros(size(bc));
dWo = zeros(size(Wo));dbo = zeros(size(bo));
dWhy = zeros(size(Why));dbhy = zeros(size(bhy));

dhnext = zeros(1, hidden_size);
dcnext = zeros(1, hidden_size);

for i = (p+seq_length - 1):-1:p
    dy = ps{i};
    dy(targets{i}) = dy(targets{i}) - 1;
    dWhy = dWhy + hs{i}' * dy;
    dbhy = dbhy + dy;
    dh =  dy * Why' + dhnext;
    dc = dcnext;
    if i == p
        [dcnext,dhnext,tdWf,tdbf,tdWi,tdbi,tdWc, tdbc, tdWo, tdbo] = lstmBackword(xs{i}, first_cprev, first_hprev, Wf, bf,Wi, bi, Wc,bc,Wo, bo, dh, dc);
    else
        [dcnext,dhnext,tdWf,tdbf,tdWi,tdbi,tdWc, tdbc, tdWo, tdbo] = lstmBackword(xs{i}, cs{i-1}, hs{i-1}, Wf, bf,Wi, bi, Wc,bc,Wo, bo, dh, dc);
    end
    dWf = dWf + tdWf;
    dbf = dbf + tdbf;
    dWi = dWi + tdWi;
    dbi = dbi + tdbi;
    dWc = dWc + tdWc;
    dbc = dbc + tdbc;
    dWo = dWo + tdWo;
    dbo = dbo + tdbo;
    
    
end

h=0.00001;
n = 1;
% Why1 = Why;
% Why2 = Why;
% Why1(n) = Why1(n)+h;Why2(n) = Why2(n)-h;
% bhy1 = bhy;
% bhy2 = bhy;
% bhy1(n) = bhy1(n) + h; bhy2(n) = bhy2(n) - h;
% Wf1 = Wf;
% Wf2 = Wf;
% Wf1(n) = Wf1(n) + h;Wf2(n) = Wf2(n) -h;
% Why = Why1;
Wf1 = Wf;
Wf2 = Wf;
Wf1(n) = Wf1(n) + h;Wf2(n) = Wf2(n) -h;
% Wi = Wi1;

% Wo1 = Wo;
% Wo2 = Wo;
% Wo1(n) = Wo1(n) + h;Wo2(n) = Wo2(n) - h;
% 
% Wo = Wo1;
% bo1 = bo;bo2 =bo;
% bo1(n) = bo1(n) + h;bo2(n)=bo2(n)-h;
% 
%  bo = bo1;
%Why = Why1;
% Wo = Wo1;
Wf = Wf1;
hprev = zeros(1,hidden_size);
cprev = zeros(1,hidden_size);
p = 1;
inputs = {}; targets = {}; xs = {}; cs = {}; hs ={}; ys = {};ps = {};
first_hprev = hprev;
first_cprev = cprev;
loss = 0;

for i = p:(p+seq_length - 1)
    inputs{i} = char_to_ix(data(i));
    targets{i} = char_to_ix(data(i+1));
    xs{i} = zeros(1,vocab_size);
    xs{i}(inputs{i}) = 1;
    [cprev,hprev] = lstmForward(xs{i},cprev,hprev,Wf,bf,Wi,bi,Wc,bc,Wo,bo);
    cs{i} = cprev;
    hs{i} = hprev;
    ys{i} = hprev * Why + bhy;
    ps{i} = exp(ys{i}) / sum(exp(ys{i}));
    loss = loss - log(ps{i}(targets{i}));
    
end

l1 = loss;
%Why = Why2;
Wf = Wf2;
% bo = bo2;
hprev = zeros(1,hidden_size);
cprev = zeros(1,hidden_size);
p = 1;
inputs = {}; targets = {}; xs = {}; cs = {}; hs ={}; ys = {};ps = {};
first_hprev = hprev;
first_cprev = cprev;
loss = 0;

for i = p:(p+seq_length - 1)
    inputs{i} = char_to_ix(data(i));
    targets{i} = char_to_ix(data(i+1));
    xs{i} = zeros(1,vocab_size);
    xs{i}(inputs{i}) = 1;
    [cprev,hprev] = lstmForward(xs{i},cprev,hprev,Wf,bf,Wi,bi,Wc,bc,Wo,bo);
    cs{i} = cprev;
    hs{i} = hprev;
    ys{i} = hprev * Why + bhy;
    ps{i} = exp(ys{i}) / sum(exp(ys{i}));
    loss = loss - log(ps{i}(targets{i}));
    
end
l2 = loss;

do = (l1-l2)/2/h;