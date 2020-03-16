from pomegranate import HiddenMarkovModel, GeneralMixtureModel, MultivariateGaussianDistribution, NormalDistribution, Kmeans
from pomegranate.utils import is_gpu_enabled, disable_gpu, enable_gpu
from pomegranate.callbacks import LambdaCallback, ModelCheckpoint
from pomegranate.io import BaseGenerator, SequenceGenerator
import numpy as np
from ticktock import tick, tock

from hmm_train_base import HMMTrainerBase, HMMInfo


def randMVG(n=40):
    '''
        generates random multivariate gaussian
    '''
    means = np.random.randn(n)
    # covs = np.random.rand(n, n)
    covs = np.eye(n, n)
    # print(covs)
    return MultivariateGaussianDistribution(means, covs)

def randGMM(n=40, nmix=5):
    '''
    generate random gaussian mixture model of multivariate distribution
    '''
    dists = [randMVG(n=n) for _ in range(nmix)]
    return GeneralMixtureModel(dists)

def randGaussian(n=40):
    '''
        generates random normal distribution
    '''
    return NormalDistribution(np.random.randint(1, 10), 1)

class PomegranateTrainer(HMMTrainerBase):
    trainerscount = 0
    def __init__(self, gpu=False, name=None, distName="G", statesNumber=3, maxIters=1000, ltr=True, nmix=5):
        '''
            ltr: is left-to-right hmm model, default to true
        '''
        super().__init__(name="PomegranateHMM", statesNumber=statesNumber, maxIters=maxIters, ltr=True)
        self.mname = name or f"PomegranateTrainer{PomegranateTrainer.trainerscount}"
        self.nmix = nmix
        self.statesNumber = statesNumber
        self.distName= distName
        self.maxIters = maxIters
        self.ltr = ltr
        self.gpu = gpu

        if (distName not in ["G", "GMM"]):
            raise TypeError(f"{distName} is not a correct dist name or not yet implemented")
        PomegranateTrainer.trainerscount += 1
        if(not self.gpu):
            disable_gpu()
            pass
        # print("is gpu enabled:", is_gpu_enabled())

    def train(self, data, lens, threads=1):
        '''
            data: observed data.
        '''
        self._buildModel(data)
        data = self._reshapeFeatures(data, lens)
        # print(data.shape, data)
        self.model.fit(data, 
            callbacks=[LambdaCallback(on_epoch_end=lambda info: info)],
            n_jobs=threads,
            # algorithm="viterbi",
            min_iterations=10,
            lr_decay=-0.5, # default is 0
            verbose=True
        )
        return self # same here, we return the self(HMMTrainer) for chaining

    def getScore(self, testData):
        return self.model.score(testData)

    def save(self, fileLocation):
        modelAsJson = self.model.to_json()
        with open(fileLocation, 'w') as saveFile:
            saveFile.write(modelAsJson)
            return fileLocation
        return None

    @staticmethod
    def load(fileLocation):
        with open(fileLocation) as modelFile:
            model = HiddenMarkovModel.from_json(modelFile.read())
            return model
        raise RuntimeError("can't load the model")

    @staticmethod
    def info(model):
        if(not isinstance(model, HiddenMarkovModel)):
            raise TypeError("model should be of type pomegranate.HiddenMarkovModel explicitly")
        return HMMInfo(model.name, model.dense_transition_matrix())
        

    def _buildModel(self, data):
        '''
        builds the model given the data to init the distributions at good point
        data: 2d matrix every row is a vector of features
        '''
        # we want to call from_matrix(transition, dists, starts, ends)
        tm = np.zeros((self.statesNumber, self.statesNumber))
        indices = [(x,x) for x in range(self.statesNumber)]
        indices.extend( [(x,x + 1) for x in range(self.statesNumber)] )
        indices.pop() # this the item (self.statesNumber-1 , self.statesNumber) that is out of bound
        indices = np.array(indices)
        tm[indices[:, 0], indices[:, 1]] = 0.5
        tm[self.statesNumber-1, self.statesNumber-1] = 0.5 # this is the end state prob, i write it alone as we may change it specificity


        dists = self._initDists(data)

        starts = np.zeros((self.statesNumber,))
        starts[0] = 1

        ends = np.zeros((self.statesNumber,))
        ends[-1] = 0.5

        self.model = HiddenMarkovModel.from_matrix(tm, dists, starts, ends, name=self.mname)
        return self.model
    
    # def _reshapeFeatures(self, origData, lens):
    #     # TODO: consider improving
    #     # input("reshaping the data")
    #     lens = np.cumsum(lens)
    #     lens = np.insert(lens, 0, 1, axis=0)
    #     res = np.array([origData[int(v - 1):int(lens[i+1] - 1)] for i,v  in enumerate(lens[:-1])])
    #     # print("res shape:", res.shape, "lens shape", len(lens), "lengths", len(lengths))
    #     # print("first x:", res[0])
    #     # for x in res:
    #     #     input(x.shape)
    #     return res

    def _initDists(self, X, distribution=MultivariateGaussianDistribution):
        # mvgd = MultivariateGaussianDistribution.from_samples(data)
        # print(mvgd)
        # gmm = GeneralMixtureModel([mvgd.copy() for _ in range(self.nmix)]) # one state dist
        # dists = [mvgd.copy() for _ in range(self.statesNumber)] #! for GHMM
        # dists = [gmm.copy() for _ in range(self.statesNumber)] #! for GMMHMM
        # dists = [randGMM(nmix=self.nmix, n=40) for _ in range(self.statesNumber)] #! for random GMMHMM
        dists = [randMVG(n=40) for _ in range(self.statesNumber)] #! for random GHMM
        return dists


        #! GMM-HMM using kmeans inits
        y = self._initkmeans(X, self.statesNumber)
        # list(map(print, y))
        return [GeneralMixtureModel.from_samples(distribution, X=X[y==i], n_components=self.nmix) for i in range(self.statesNumber)]


        #! Kmeans init
        if not isinstance(X, BaseGenerator):
            data_generator = SequenceGenerator(X, None, None)
        else:
            data_generator = X

        initialization_batch_size = len(data_generator)

        X_ = []
        data = data_generator.batches()
        for i in range(initialization_batch_size):
            batch = next(data)
            X_.extend(batch[0])

        X_concat = np.concatenate(X_)
        if X_concat.ndim == 1:
            X_concat = X_concat.reshape(X_concat.shape[0], 1)
        n, d = X_concat.shape
        clf = Kmeans(self.statesNumber, init="kmeans++", n_init=1) # init should be one of
        clf.fit(X_concat, max_iterations=None, batches_per_epoch=None)
        y = clf.predict(X_concat)
        if callable(distribution):
            if d == 1:
                dists = [distribution.from_samples(X_concat[y == i][:,0]) 
                    for i in range(self.statesNumber)]
            elif distribution.blank().d > 1:
                dists = [distribution.from_samples(X_concat[y == i]) 
                for i in range(self.statesNumber)]
            else:
                print("error")
        return dists

    def _initkmeans(self, X, numClasses):
        clf = Kmeans(numClasses, init="kmeans++", n_init=1) # init should be one of
        clf.fit(X, max_iterations=1, batches_per_epoch=None)
        y = clf.predict(X)
        # return GeneralMixtureModel([MultivariateGaussianDistribution.from_samples(X[y == i]) for i in range(self.nmix)])
        return y
    def _reshapeFeatures(self, origData, lens):
        lens = np.cumsum(lens)
        lens = np.insert(lens, 0, 0, axis=0)
        return np.array([origData[int(v):int(lens[i+1])] for i,v  in enumerate(lens[:-1])])

