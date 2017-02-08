load data;
chars = unique(data);
data_size = length(data);
vocab_size = length(chars);

hidden_size = 100;
seq_length = 25;
learning_rate = 0.1;
epoch = 1000000;

char_to_ix = containers.Map();
for i=1:vocab_size
    char_to_ix(chars(i)) = i;
end
ix_to_char = chars;
Wf = random('norm',0,1,[(vocab_size + hidden_size), hidden_size]) / sqrt(hidden_size);
bf = random('norm',0,1,[1,hidden_size]) / sqrt(hidden_size);
Wi = random('norm',0,1,[(vocab_size + hidden_size), hidden_size]) / sqrt(hidden_size);
bi = random('norm',0,1,[1,hidden_size]) / sqrt(hidden_size);
Wc = random('norm',0,1,[(vocab_size + hidden_size), hidden_size]) / sqrt(hidden_size);
bc = random('norm',0,1,[1,hidden_size]) / sqrt(hidden_size);
Wo = random('norm',0,1,[(vocab_size + hidden_size), hidden_size]) / sqrt(hidden_size);
bo = random('norm',0,1,[1,hidden_size]) / sqrt(hidden_size);
Why = random('norm',0,1,[hidden_size, vocab_size]) / sqrt(vocab_size);
bhy = random('norm',0,1,[1, vocab_size]) / sqrt(vocab_size);
n = 0;p = 1;
mWf = zeros(size(Wf));mbf = zeros(size(bf));
mWi = zeros(size(Wi));mbi = zeros(size(bi));
mWc = zeros(size(Wc));mbc = zeros(size(bc));
mWo = zeros(size(Wo));mbo = zeros(size(bo));

mWhy = zeros(size(Why));mbhy = zeros(size(bhy));

hprev = zeros(1,hidden_size);
cprev = zeros(1,hidden_size);
for epoch_i = 1:epoch
    if (p + seq_length >= data_size) || (n == 0)
        hprev = zeros(1,hidden_size);
        cprev = zeros(1,hidden_size);
        p = 1;
    end
    
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
    
    if mod(epoch_i,100) == 0
        display(epoch_i);display(loss);
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
    mWf =  mWf + dWf .* dWf;
    mbf = mbf + dbf .* dbf;
    mWi = mWi + dWi .* dWi;
    mbi = mbi + dbi .* dbi;
    mWc = mWc + dWc .* dWc;
    mbc = mbc + dbc .* dbc;
    mWo = mWo + dWo .* dWo;
    mbo = mbo + dbo .* dbo;
    mWhy = mWhy + dWhy .* dWhy;
    mbhy = mbhy + dbhy .* dbhy;
      
    Wf = Wf - learning_rate * dWf ./ sqrt(mWf + 1e-8);
    bf = bf - learning_rate * dbf ./ sqrt(mbf + 1e-8);
    Wi = Wi - learning_rate * dWi ./ sqrt(mWi + 1e-8);
    bi = bi - learning_rate * dbi ./ sqrt(mbi + 1e-8);
    Wc = Wc - learning_rate * dWc ./ sqrt(mWc + 1e-8);
    bc = bc - learning_rate * dbc ./ sqrt(mbc + 1e-8);
    Wo = Wo - learning_rate * dWo ./ sqrt(mWo + 1e-8);
    bo = bo - learning_rate * dbo ./ sqrt(mbo + 1e-8);
    Why = Why - learning_rate * dWhy ./ sqrt(mWhy + 1e-8);
    bhy = bhy - learning_rate * dbhy ./ sqrt(mbhy + 1e-8);
    p = p + seq_length;
    n = n + 1;
end

save Wf
save bf
save Wi
save bi
save Wc
save bc
save Wo
save bo
save Why
save bhy
save char_to_ix
save ix_to_char