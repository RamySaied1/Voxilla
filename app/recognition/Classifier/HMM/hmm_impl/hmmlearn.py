import numpy as np
from hmmlearn import hmm

from hmm_train_base import HMMTrainerBase

class HMMLearnTrainer(HMMTrainerBase):
    def __init__(self, name="", GMM=False, statesNumber=3, maxIters=1000, ltr=True, startAndEndSates=False):
        '''
            ltr: is left-to-right hmm model, default to true
        '''
        self.n_states = statesNumber
        self.GMM= GMM
        self.maxIters = maxIters
        self.ltr = ltr
        self.ses = startAndEndSates
        self.n = self.n_states + 1 if self.ses else self.n_states
        
        # initParams = "cm" if ltr else "cmst"
        if (self.ltr):
            self.model = hmm.GMMHMM(verbose=True, n_mix=5, n_components=self.n, n_iter=self.maxIters, init_params="cm", params="cmt", covariance_type="diag") if self.GMM else hmm.GaussianHMM(verbose=True, n_components=self.n, n_iter=self.maxIters, init_params="cm", params="cmt", covariance_type="diag")
        else:
            self.model = hmm.GMMHMM(verbose=True, n_mix=5, n_components=self.n, n_iter=self.maxIters) if self.GMM else hmm.GaussianHMM(verbose=True, n_components=self.n, n_iter=self.maxIters)

        if (self.ltr and self.ses):
            n = self.n

            sp = np.zeros((n,))
            sp[0] = 1
            # print(sp)
            self.model.startprob_ = sp

            tm = np.zeros((n, n))
            indices = [(x,x) for x in range(n)]
            indices.extend( [(x,x + 1) for x in range(n)] )
            indices.pop() # this the item (n-1 , n) that is out of bound
            indices = np.array(indices)
            tm[indices[:, 0], indices[:, 1]] = 0.5
            tm[0, 1] = 1.0 # transition from start state to S0
            tm[0, 0] = 0.0 # start state has no self loops
            tm[n-1, n-1] = 1.0
            self.model.transmat_ = tm
            print(tm)
        elif (self.ltr):
            n = self.n_states

            sp = np.zeros((n,))
            sp[0] = 1
            # print(sp)
            self.model.startprob_ = sp

            tm = np.zeros((n, n))
            indices = [(x,x) for x in range(n)]
            indices.extend( [(x,x + 1) for x in range(n)] )
            indices.pop() # this the item (n-1 , n) that is out of bound
            indices = np.array(indices)
            tm[indices[:, 0], indices[:, 1]] = 0.5
            tm[n-1, n-1] = 1.0
            self.model.transmat_ = tm
            print(tm)
        self.model.name = name


    def train(self, data, lens=None):
        '''
            data: observed data.
        '''
        _ = self.model.fit(data, lengths=lens) if lens else self.model.fit(data) # fit returns the object itself for chaining
        return self # same here, we return the self(HMMTrainer) for chaining

    def getScore(self, testData):
        return self.model.score(testData)
