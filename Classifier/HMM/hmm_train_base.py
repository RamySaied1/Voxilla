class HMMTrainerBase:
    def __init__(self, name=None, statesNumber=3, maxIters=1000, ltr=True):
        super().__init__()
        self.name = None
        self.model = None


    def train(self):
        pass

    def score(self):
        pass
    
    def save(self):
        pass

    def load(self):
        pass

    @property
    def internalModel(self):
        return self.model



class HMMInfo(object):
    def __init__(self, name, transmat):
        self.name = name
        self.transmat = transmat