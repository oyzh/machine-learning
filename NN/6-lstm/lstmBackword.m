function [dcprev,dhprev,dWf,dbf,dWi,dbi,dWc, dbc, dWo,dbo] = lstmBackword(x, cprev, hprev, Wf, bf,Wi, bi, Wc,bc,Wo, bo, dht, dct)
% lstm cell, lstm input is x , cprev, hprev. output is ct and ht
% input
% x - 1*m, current input
% cprev - 1*n, prev cell memory
% hprev - 1*n, prev cell memory
% Wf - (m+n) * n,
% br - 1*n,
% Wi - (m+n) * n,
% bi - 1*n,
% Wc - (m+n) * n,
% bc - 1*n,
% Wo - (m+n) * n,
% bo - 1*n
% this version I will caculate twice, to be best is not.
	input_concat = [hprev, x];
	t1 = input_concat*Wf+bf;
	ft = sigmoid(t1); %(1*n)
	t2 = input_concat*Wi + bi;
	it = sigmoid(t2); %(1*n)
	t3 = input_concat * Wc + bc;
	ct_hat = tanh(t3);%(1*n)
	ct = ft .* cprev + it .* ct_hat; %(1*n)
	t4 = input_concat * Wo + bo;
	ot = sigmoid(t4);%(1*n)
	%ht = ot .* tanh(ct);%(1*n)

	dinput_concat = zeros(size(input_concat));
	dct = dct + dht .* ot .* dtanh(ct);
	Dot = dht .* tanh(ct);

	dt4 = Dot .* dsigmoid(t4);
	dbo = dt4;
	dWo = input_concat' * dt4;
	dinput_concat = dinput_concat + dt4 * Wo';

	dft = dct .* cprev;
	dcprev = dct .* ft;
	dit = dct .* ct_hat;
	dct_hat = dct .* it;

	dt3 = dct_hat .* dtanh(t3);
	dbc = dt3;
	dWc = input_concat' * dt3;
	dinput_concat = dinput_concat +  dt3 * Wc';

	dt2 = dit .* dsigmoid(t2);
	dbi = dt2;
	dWi = input_concat' * dt2;
	dinput_concat = dinput_concat + dt2 * Wi';

	dt1 = dft .* dsigmoid(t1);
	dbf = dt1;
	dWf = input_concat' * dt1;
	dinput_concat = dinput_concat + dt1 * Wf';

	dhprev = dinput_concat(1:length(hprev));
	
end
