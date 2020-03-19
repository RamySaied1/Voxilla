import os
from my_mfcc import mfcc
# from python_speech_features import mfcc
# from pydub import AudioSegment
import numpy as np
import pickle
# from Classifier.HMM.HMMTrainer import HMMTrainer
from hmm_impl.pomegranate import PomegranateTrainer
from align_40_phones import main as runAlign
from ticktock import tick, tock
from main_base import HMMBase

class HMM_POM(HMMBase):
	'''
		An interface for training A hmm of phones using pomegranate implementation
	'''
	def __init__(self,
			librispeechDir="data\\train-clean-100",
			alignmentsDir = "data/alignments",
			trainDir = "data/alignments",
			testDir = "data/alignments",
			normalizationDir = "data/normalization",
			featuresDir = "data/models-features",
			modelsDir = "data/models",
			ext_model = ".model.json",
			verbose=False, normalize=True, n_skip=0,
			gpu=False, threads=1, GMM=False
		):
		super().__init__(
			librispeechDir = librispeechDir,
			alignmentsDir = alignmentsDir,
			trainDir = trainDir,
			testDir = testDir,
			normalizationDir = normalizationDir,
			featuresDir = featuresDir,
			modelsDir = modelsDir,
			ext_model = ext_model,
			n_skip = n_skip, verbose = verbose, normalize = normalize
		)
		self.gpu = gpu
		self.threads = threads
		self.GMM = GMM

	#!
	def _loadModel(self, loc):
		'''
			load the model from io in loc
		'''
		return PomegranateTrainer.load(loc)

	def _saveModel(self, loc, model):
		'''
			saves the model to the given location
		'''
		modelAsJson = model.to_json()
		with open(loc, 'w') as saveFile:
			saveFile.write(modelAsJson)
			return True
		return False

	def _trainModel(self, label, data):
		'''
			train single model of label using data
			data is tuple of (features, lengths)
		'''
		trainer = PomegranateTrainer(name=label, gpu=self.gpu)
		return trainer.train(data[0], lens=data[1], threads=self.threads).model

	def _modelInfo(self, model):
		'''
			returns the model info of the given model
		'''
		return PomegranateTrainer.info(model)

	def _computeScore(self, model, data):
		features, lengths = data
		numberOfSamples = len(lengths)
		lengths = np.cumsum(lengths)
		lengths = np.insert(lengths, 0, 0, axis=0)
		# print(features.shape)
		features = np.array( [model.log_probability(features[int(v):int(lengths[i+1])]) for i,v  in enumerate(lengths[:-1])] )
		# print(len(features), "==>", end=" ")
		features = [f for f in features if f != -np.inf]
		# print(len(features))
		if (len(features) <= 0.5 * numberOfSamples):
			# print(len(features), numberOfSamples)
			print(f"computeScore: many -inf values from {model.name}")
			return -np.inf
		# print((sum(features) / len(features)))
		return sum(features)

	def _generateSamples(self, numSamples, model):
		sample, path = model.sample(path=True)
		path = list( map(lambda state:state.name, path) )
		print("path is", path)
		# print(type(samples), "shape of the samples:", samples.shape)
		self._verbose("sample is", sample)
		print("taking this sample and compute the prob of it on the model")
		logprob = model.log_probability(sample)
		prob = model.probability(sample)
		print(logprob, prob)
		return sample

if __name__ == "__main__":
	from fire import Fire
	tick("timing the whole run")
	Fire(HMM_POM)
	tock("the whole run")