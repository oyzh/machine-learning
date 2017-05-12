import sys
sys.path.append("../CART")

from cart import *
import numpy as np

def res(model, data, label):
	N,_ = data.shape

	r = np.zeros_like(label)
	for i in xrange(N):
		y_i = model(data[i,:])
		r[i] = label[i] - y_i
	return r

def loss(model, data, label):
	N,_ = data.shape

	l = 0
	for i in xrange(N):
		l = l + (model(data[i,:]) - label[i])**2
	return l

def init_model(c):
	return lambda x : c

def update_model(model, data, r):
	new_tree = cart_tree(data, r)
	new_model = lambda x : model(x) + run_cart(new_tree,x)
	return new_model

if __name__ == '__main__':
	data, label = generate_data(data_num, data_dim)
	new_label = np.zeros_like(label)

	c = np.sum(label) / data_num
	model = init_model(c)

	for i in xrange(10):
		l = loss(model, data, label)
		print "epoch ",i,":",l
		r = res(model, data, label)
		model =  update_model(model, data, r)
