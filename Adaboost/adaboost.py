# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *

data_num1 = 200;
data_num2 = 200;
data_num = data_num1+data_num2;

def sample_data(data,label, weight, N):
    """
    data is (M*2),sample N rows according to weight
    weight is also (M*1)
    sum of weight is 1.
    """
    M,_ = data.shape;
    
    index = np.random.choice(M, N, weight.data)
    return data[index,:],label[index]
    
def add_new_f(old_f, alpha,  f):
    return lambda x: old_f(x) + alpha * f(x)
    
def select_f(data, label):
	"""
	use the liner classification
	A*[w1,w2,b] = B,solve the liner equation
	the data must be 2-d input,but it's easy to solve high dim
	"""
	A = np.zeros([3,3])
	A[0,0] = np.sum(data[:,0]**2)
	A[0,1] = np.sum(np.dot(data[:,0],data[:,1]))
	A[0,2] = np.sum(data[:,0])
	A[1,0] = A[0,1]
	A[1,1] = np.sum(data[:,1]**2)
	A[1,2] = np.sum(data[:,1])
	A[2,0] = A[0,2]
	A[2,1] = A[1,2]
	A[2,2] = label.shape[0]

	B = np.zeros([3])
	B[0] = np.sum(np.dot(data[:,0], label))
	B[1] = np.sum(np.dot(data[:,1], label))
	B[2] = np.sum(label)

	[w1,w2,b] = np.linalg.solve(A, B)

	return lambda x: np.sign(w1*x[:,0] + w2*x[:,1]+b)
    
if __name__ == '__main__':
    data, label = generate_data(data_num,data_num1,data_num2)

    weight = np.ones(data_num)*(1./data_num);

    def f(x):
        return 0;

    plt.figure(figsize=(10,10))
    err = []

    for i in range(200):# the main iterate, some rules should be added to end the loop
        s_data, s_label = sample_data(data, label, weight, data_num)
        new_f = select_f(s_data, s_label)
        e = np.sum((new_f(data)!=label)*weight)
        alpha = .5*np.log((1-e) / e)
        weight = weight * np.exp(-alpha * label * new_f(data))
        Z = np.sum(weight)
        weight = weight / Z
        f = add_new_f(f, alpha, new_f)

        index = (np.sign(f(data)) == label)
        correct_rate = np.sum(index) / float(data_num)
        err = [err,1-correct_rate]
        print correct_rate

    index = (np.sign(f(data)) == label)
    plt.plot(data[label==-1,0],data[label==-1,1],label='class 1',ls='none',marker='o',color='red',linewidth=4)
    plt.plot(data[label==1,0],data[label==1,1],label='class 2',ls='none',marker='s',color='green',linewidth=4)
    plt.plot(data[~index,0],data[~index,1],label = 'classfiy error',ls='none',marker='v',color='black',linewidth=2)
    plt.legend()
    plt.title('Adaboost test')
    plt.show(1)

