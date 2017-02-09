import numpy as np
import matplotlib.pyplot as plt

def generate_data(N, dim):
    """
    generate some random data
    """
    data = np.random.sample((N,dim))
    label = np.sin(np.sum(data,axis=1))
    return data,label

def get_var_split(data,label):
    """
    find the best feature and split point j and s,
    according to the minize the cost function.
    """
    data_n = data.shape[0]
    feature_n = data.shape[1]
    min_j,min_s,min_v = 0,0,float("inf")
    min_c1,min_c2 = 0, 0;

    for j in xrange(feature_n):
        feature_j = data[:,j]
        point = list(set(feature_j))
        point.sort()
        for s in point:
            R1_index = (feature_j <= s)
            R2_index = (feature_j > s)
            if np.sum(R1_index) == 0:
                c1 = 0
            else:
                c1 = np.mean(label[R1_index])
            if np.sum(R2_index) == 0:
                c2 = 0
            else:
                c2 = np.mean(label[R2_index])
            value = np.sum((label[R1_index] - c1)**2) + np.sum((label[R2_index] - c2)**2)
            if value < min_v:
                min_v,min_j,min_s = value,j,s
                min_c1,min_c2 = c1,c2

    return min_j,min_s,min_c1,min_c2

data_num = 10000
data_dim = 2
result_number = 10

class Tree():
    """
    the node of the cart Tree
    """
    j,s,c1,c2 = 0,0,0,0
    left,right = 0,0

def cart_tree(data, label):
    """
    build the Tree by recursive,split the Data by j and s.
    """
    node = Tree()
    node.j,node.s,node.c1,node.c2 = get_var_split(data,label)
    feature_j = data[:,node.j]
    D1 = data[feature_j<=node.s,:]
    L1 = label[feature_j<=node.s]
    D2 = data[feature_j>node.s,:]
    L2 = label[feature_j>node.s]
    if D1.shape[0] > result_number:
        node.left = cart_tree(D1,L1)
    if D2.shape[0] > result_number:
        node.right = cart_tree(D2,L2)
    return node


def run_cart(node, features):
    """
    find the value according to the tree(node) and data(features).
    """
    if features[node.j] <= node.s:
        if node.left == 0:
            return node.c1
        else:
            return run_cart(node.left,features)
    if features[node.j] > node.s:
        if node.right == 0:
            return node.c2
        else:
            return run_cart(node.right,features)

if __name__ == '__main__':
    data,label = generate_data(data_num, data_dim)
    #data = np.arange(0,np.pi,0.01)
    #data = data.reshape([data.shape[0],1])
    #label = np.sin(data)
    #    plt.scatter(data[:,0],data[:,1],c=label,alpha=0.5)
    #    plt.show()
    root = cart_tree(data,label)


    new_label = np.zeros_like(label)
    for i in xrange(data.shape[0]):
        new_label[i] = run_cart(root,data[i,:])

    plt.scatter(data[:,0],data[:,1],c=new_label,alpha=0.5)
    #plt.plot(data[:,0],label)
    plt.show()
