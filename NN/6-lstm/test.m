m = 768; % input number
n = 200; % hiden number

x = rand(1,m);
cprev = rand(1,n);
hprev = rand(1,n);
Wf = rand((m+n), n) / n;
bf = rand(1,n) / n;
Wi = rand((m+n), n) / n;
bi = rand(1,n) / n;
Wc = rand((m+n), n) / n;
bc = rand(1,n) / n;
Wo = rand((m+n), n) / n;
bo = rand(1,n) / n;
[ct,ht] = lstmForward(x,cprev,hprev,Wf,bf,Wi,bi,Wc,bc,Wo,bo);
dct = ones(size(ct));
dht = ones(size(ht));
[dcprev,dhprev,dWf,dbf,dWi,dbi,dWc, dbc, dWo,dbo] = lstmBackword(x, cprev, hprev, Wf, bf,Wi, bi, Wc,bc,Wo, bo, dht, dct);
