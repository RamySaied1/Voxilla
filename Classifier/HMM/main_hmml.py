import os
from my_mfcc import mfcc
# from python_speech_features import mfcc
# from pydub import AudioSegment
import numpy as np
import pickle
# from Classifier.HMM.HMMTrainer import HMMTrainer
from hmm_impl.hmmlearn import HMMLearnTrainer as HMMTrainer
from align_40_phones import main as runAlign
from ticktock import tick, tock

from main_base import HMMBase
from hmm_train_base import HMMInfo

class HMM_HMML(HMMBase):
	'''
		An interface for training A hmm of phones using hmmlearn implementation
	'''
	def __init__(self, GMM=False,
			librispeechDir="data/train-clean-100",
			alignmentsDir = "data/alignments",
			trainDir = "data/alignments",
			testDir = "data/alignments",
			normalizationDir = "data/normalization",
			featuresDir = "data/models-features",
			modelsDir = "data/models",
			verbose=False, normalize=True, n_skip=0
		):
		super().__init__(
			librispeechDir = librispeechDir,
			alignmentsDir = alignmentsDir,
			trainDir = trainDir,
			testDir = testDir,
			normalizationDir = normalizationDir,
			featuresDir = featuresDir,
			modelsDir = modelsDir,
			ext_model=".model.pkl",
			n_skip = n_skip, verbose = verbose, normalize = normalize
		)
		self.GMM = GMM

	def modelsMonitor(self, *phones, modelsSet=200):
		models = self._loadModels(*phones, path=self._getModelsPath(self.modelsDir, modelsSet))
		for model in models:
			print(model.name, ":", model.monitor_, "converged" if model.monitor_.converged else "not converged")

	def generateSample(self, *phones, numSamples=1, modelsSet=200):
		models = self._loadModels(*phones, path=self._getModelsPath(self.modelsDir, modelsSet))
		for model in models:
			print(f"for model {model.name}, generating {numSamples} samples")
			samples = model.sample(n_samples=numSamples)
			# print("shape of the samples:", samples.shape)
			print(type(samples), len(samples))
			self._verbose(samples)
			firstSample = samples[0]
			logprob = model.score(firstSample)
			print(logprob)
			
	#!
	def _loadModel(self, loc):
		'''
			load the model from io in loc
			loc contains the info saved by saveModel function
		'''
		with open(loc, "rb") as file:
			phoneHMM = pickle.load(file)
			# phoneHMM.name = os.path.basename(loc).replace(self.ext_model, "")
			return phoneHMM
		return False

	def _saveModel(self, loc, model):
		'''
			saves the model to the given location
			model is the object returned from the trainModel
		'''
		with open(loc, 'wb') as saveFile:
			label = model.name
			pickle.dump(model, saveFile)
			self._verbose(f"model of {label} saved in {os.path.abspath(loc)}")
			return True
		return False

	def _trainModel(self, label, data):
		'''
			train single model of label using data
			data is tuple of (features, lengths)
		'''
		hmml = HMMTrainer(GMM=self.GMM, name=label)
		hmml.train(data[0], lens=data[1])
		return hmml.model

	def _computeScore(self, model, data):
		'''
			computes the score of the data given the model. the max-liklihood of generating the data from this model
		'''
		features, lens = data
		return model.score(features, lengths=lens)

	def _modelInfo(self, model):
		'''
			returns the model info of the given model
		'''
		# TODO consider using dict as it is more convenient and easier
		# return {
		# 	"name": model.name,
		# 	"transmat": model.transmat_
		# }
		return HMMInfo(model.name, transmat=model.transmat_)

if __name__ == "__main__":
	from fire import Fire
	tick("timing the whole run")
	Fire(HMM_HMML)
	tock("the whole run")