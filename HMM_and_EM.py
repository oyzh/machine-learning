import numpy as np
import matplotlib.pyplot as plt

class Model():
    A = np.array([[0.5,0.2,0.3],
                  [0.3,0.5,0.2],
                  [0.2,0.3,0.5]])

    B = np.array([[0.5,0.5],
                  [0.4,0.6],
                  [0.7,0.3]])

    pi = np.array([0.2,0.4,0.4])

    def empty(self):
        self.A = np.zeros_like(self.A)
        self.B = np.zeros_like(self.B)
        self.pi = np.zeros_like(self.pi)

    def random(self):
        """
        TODO, random A,B and pi
        """
        pass
def generate_data(trainModel,L):
    """
    (A,B,pi) is the hidden markov model
    A -> state trans matrix, N*N
    B -> observation probality matrix, N*M
    pi -> the init state probality vector, 1*N
    L -> the length of the series we sample
    """
    A,B,pi = trainModel.A,trainModel.B,trainModel.pi
    [N,M] = B.shape
    q = np.zeros(L)
    o = np.zeros(L)

    p = pi
    for i in xrange(L):
        q[i] = np.random.choice(N,1,p=p)
        o[i] = np.random.choice(M,1,p=B[q[i],:])
        p = A[q[i],:]

    return q,o

def get_probability_forward(M, o):
    """
    given the HMM model M, and the output o, return the probalibility
    of o, using the forward algorithm
    o is L length
    return P(o|M)
    """
    A,B,pi = M.A,M.B,M.pi
    L = o.shape[0]
    [N,_] = B.shape # N is the number of state,M is number of observation

    # init alpha, at the time t, prev state is o1,o2,..ot,the probability of each state is alpha[i]
    alpha = np.array([pi[i]*B[i,o[0]] for i in xrange(N)])

    next_alpha = np.zeros_like(alpha)
    for i in xrange(1,L):
        for j in xrange(N):
            next_alpha[j] = sum([alpha[s]*A[s,j] for s in xrange(N)]) * B[j,o[i]]
        alpha = np.copy(next_alpha) # do not direct copy!!

    return np.sum(alpha)

def get_probability_backward(M, o):
    """
    given the HMM model M, and the output o, return the probalibility
    of o, using the backward algorithm
    """
    A,B,pi = M.A, M.B, M.pi
    L = o.shape[0]
    [N,_] = B.shape

    beta = np.ones(N)

    T = range(L-1)
    T.reverse()
    prev_beta = np.zeros_like(beta)
    for i in T:
        for j in xrange(N):
            prev_beta[j] = np.sum([A[j,s]*B[s,o[i+1]]*beta[s] for s in xrange(N)])
        beta = np.copy(prev_beta)

    return np.sum([pi[i]*B[i,o[0]]*beta[i] for i in xrange(N)])

def supervised_learning(model , Q, O):
    """
    supervised learning algorithm using the hidden state information and output information
    model -> we should have a empty model, we assume that length of each  Q and O are equal
    Q -> the series of state S*L matrix
    N -> obesevation S*L matrix
    that's say we have S sample, and each sample is L length
    we can direct find the A,B and pi
    """

    [S,L] = Q.shape
    [N,M] = model.B.shape

    HA = np.zeros_like(model.A, dtype=float) #save the number of state trans
    for s in xrange(S):
        for l in xrange(L-1):
            HA[Q[s,l],Q[s,l+1]] = HA[Q[s,l],Q[s,l+1]] + 1
    model.A = HA / np.sum(HA,axis=1)

    HB = np.zeros_like(model.B, dtype=float) #save the number of state to observation number
    for s in xrange(S):
        for l in xrange(L):
            HB[Q[s,l],O[s,l]] = HB[Q[s,l], O[s,l]] + 1
    model.B = HB / np.reshape(np.sum(HB,axis=1), [N,1])

    for s in xrange(S):
        model.pi[Q[s,0]] = model.pi[Q[s,0]] + 1
    model.pi = model.pi / np.sum(model.pi)
    return model


class Baum_Welch():
    """
    implement Baum Welch algorithm
    """
    def __init__(self, model, O, epochs=100):
        self.model = model
        self.O = O
        self.N,self.M = model.B.shape
        self.S,self.T = O.shape
        self.epochs = epochs

    def get_alpha(self):
        alpha = np.zeros([self.S, self.T, self.N])
        for s in xrange(self.S):
            alpha_s = alpha[s,:,:]
            O_s = self.O[s,:]
            alpha_s[0,:] = self.model.pi * self.model.B[:,O_s[0]]
            for i in xrange(1,self.T):
                for j in xrange(self.N):
                    alpha_s[i,j] = np.sum([alpha_s[i-1,k]*self.model.A[k,j] for k in xrange(self.N)]) * self.model.B[j,O_s[i]]
        return alpha

    def get_beta(self):
        beta = np.zeros([self.S, self.T, self.N])
        for s in xrange(self.S):
            beta_s = beta[s,:,:]
            O_s = self.O[s,:]
            beta_s[self.T-1,:] = 1
            for i in reversed(xrange(self.T-1)):
                for j in xrange(self.N):
                    beta_s[i,j] = np.sum([self.model.A[j,k]*self.model.B[k,O_s[i+1]]*beta_s[i+1,k] for k in xrange(self.N)])
        return beta

    def get_gama(self,alpha, beta):
        gama = np.zeros([self.S, self.T, self.N])
        for s in xrange(self.S):
            gama_s = gama[s,:,:]
            alpha_s = alpha[s,:,:]
            beta_s = beta[s,:,:]
            for t in xrange(self.T):
                p = alpha_s[t,:]*beta_s[t,:]
                gama_s[t,:] = p / np.sum(p)

        return gama

    def get_eta(self, alpha, beta):
        eta = np.zeros([self.S, self.T-1, self.N, self.N])
        for s in xrange(self.S):
            eta_s = eta[s,:,:,:]
            alpha_s = alpha[s,:,:]
            beta_s = beta[s,:,:]
            O_s  = self.O[s,:]
            for t in xrange(self.T-1):
                p = alpha_s[t,:]*self.model.A*self.model.B[:,O_s[t+1]]*beta_s[t+1,:]
                eta_s[t,:,:] = p / np.sum(p)

        return eta

    def update(self, gama, eta):
        sum_eta = np.sum(eta,axis=(0,1))
        sum_gama = np.sum(gama[:,:-1,:], axis=(0,1))
        self.model.A = sum_eta / sum_gama
        self.model.pi = np.mean(gama[:,0,:],axis=0)

    def __run_epoch__(self):
        """
        using only output data, we shold using EM algorithm and Baum-Welch to train the model
        """
        # step1, caculate alpha:S*T*N, beta:S*T*N
        alpha = self.get_alpha()
        beta = self.get_beta()
        # step2, caculate gama:S*T*N, eta:S*(T-1)*N*N
        gama = self.get_gama(alpha, beta)
        eta = self.get_eta(alpha, beta)
        # step3, caculate new A,B,pi
        self.update(gama, eta)

    def train(self):
        for epoch in xrange(self.epochs):
            print "epoch", epoch
            self.__run_epoch__()
        return self.model

def predict_appro(model, o):
    """
    according to HMM model and obesevation, predict the most possibal state series.
    using the approximate algorithm.
    return q_predict -> length equal to o
    """
    q = np.zeros_like(o)

    return q

def predict_Viterbi(model, o):
    """
    like predict_appro, but using the Viterbi algorithm.
    """
    q = np.zeros_like(o)

    return q

if __name__ == '__main__':
    realModel = Model()
    train_S = 20 # train samples
    train_L = 1000 # sample's length

    train_q = np.zeros([train_S,train_L])
    train_o = np.zeros([train_S,train_L])
    for i in xrange(train_S):
        train_q[i,:],train_o[i,:] = generate_data(realModel, train_L)

    p_forward = get_probability_forward(realModel, np.array([0,1,0]))
    p_backward = get_probability_backward(realModel, np.array([0,1,0]))

    #q_predict_appro = predict_appro(realModel, o_train)
    #q_predict_Viterbi = predict_Viterbi(realModel, o_train)

    emptyModel = Model()
    emptyModel.empty()
    sl_Model = supervised_learning(emptyModel, train_q, train_o)
    #print sl_Model.A
    #print sl_Model.B
    #print sl_Model.pi

    emptyModel.random()
    epochs = 5
    BW = Baum_Welch(emptyModel, train_o, epochs)
    BW_model = BW.train()
