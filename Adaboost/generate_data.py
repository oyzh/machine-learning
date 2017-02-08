import numpy as np

def generate_data1(N):
    r = np.random.random_sample((N)) * 5
    theta = np.random.random_sample((N)) * 2 * np.pi
    x = r * np.cos(theta) + 10;
    y = r * np.sin(theta) + 10;
    return x,y

def generate_data2(N):
    r = np.random.random_sample((N))*5 + 5
    theta = np.random.random_sample((N)) * 2 * np.pi
    x = r * np.cos(theta) + 10
    y = r * np.sin(theta) + 10;
    return x,y

def generate_data(data_num,data_num1,data_num2):
    """
    generate training data.
    return (N*2) inputs and (N*1) label.
    """
    x1,y1 = generate_data1(data_num1)
    x2,y2 = generate_data2(data_num2)
    X = np.array([x1,x2]).reshape(data_num)
    Y  = np.array([y1,y2]).reshape(data_num)
    L = np.ones_like(X)
    L[range(data_num1)] = -1
    return np.array([X,Y]).T, L
