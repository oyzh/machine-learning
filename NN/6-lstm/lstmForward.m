function [ct,ht] = lstmForward(x, cprev, hprev, Wf, bf,Wi, bi, Wc, bc, Wo, bo)
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
% we'd best save the inter result, I don't save it for lazy..
	input_concat = [hprev, x];
	ft = sigmoid(input_concat * Wf + bf);
	it = sigmoid(input_concat * Wi + bi);
	ct_hat = tanh(input_concat * Wc + bc);
	ct = ft .* cprev + it .* ct_hat;
	ot = sigmoid(input_concat * Wo + bo);
	ht = ot .* tanh(ct);
end
