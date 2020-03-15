'''
    This file can be rewritten using different other than hmmlean, or may be also implemented from scratch
'''
from hmmlearn import hmm
import numpy as np
class HMMTrainer(object):
    def __init__(self, distName="GaussianHMM", statesNumber=3, maxIters=1000, ltr=True):
        '''
            ltr: is left-to-right hmm model, default to true
        '''
        self.n_states = statesNumber
        self.distName= distName
        self.maxIters = maxIters
        self.ltr = ltr

        if (distName not in ["GaussianHMM",]):
            raise TypeError(f"{distName} is not a correct dist name or not yet implemented")
        
        # initParams = "cm" if ltr else "cmst"
        if (self.ltr):
            self.model = hmm.GaussianHMM(n_components=self.n_states, n_iter=self.maxIters, init_params="cm", params="cmt", covariance_type="diag")
        else:
            self.model = hmm.GaussianHMM(n_components=self.n_states, n_iter=self.maxIters)

        if (self.ltr):
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
            # print(tm)


    def train(self, data, lens=None):
        '''
            data: observed data.
        '''
        _ = self.model.fit(data, lengths=lens) if lens else self.model.fit(data) # fit returns the object itself for chaining
        return self # same here, we return the self(HMMTrainer) for chaining

    def getScore(self, testData):
        return self.model.score(testData)