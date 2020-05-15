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
			librispeechDir="data/train-clean-100",
			alignmentsDir = "data/alignments",
			trainDir = "data/alignments",
			testDir = "data/alignments",
			normalizationDir = "data/normalization",
			featuresDir = "data/models-features",
			modelsDir = "data/models",
			ext_model = ".model.json",
			ext_emissions = ".emis.logprob.txt",
			emissionsDir = None, # None means the same as .feat file
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
		self.ext_feat = ".feat"
		self.ext_emissions = ext_emissions
		self.emissionsDir = emissionsDir

	def emissions(self, *phones, path=None, modelsSet=200):
		'''
			extract emissions probabilities of a file or all files in the dir of path is a dir
		'''
		if(not os.path.exists(path)):
			raise FileNotFoundError("The path can't be found")
		paths = [path]
		if(os.path.isdir(path)):
			self._verbose(f"extracting emissions of dir {os.path.abspath(path)}")
			join = lambda f: os.path.join(path, f)
			exist = lambda f: os.path.exists(join(f.replace(self.ext_feat, self.ext_emissions)))
			paths = [join(file) for file in sorted(os.listdir(path)) if file.endswith(self.ext_feat) and not exist(file)]
		for featFile in paths:
			tick()
			self._fileEmissions(*phones, featPath=featFile, modelsSet=modelsSet)
			tock()

	def _fileEmissions(self, *phones, featPath=None, modelsSet=200):
		if(featPath == None):
			raise TypeError("fpath can't be None")
		from FeatureExtraction.htk_featio import read_htk_user_feat as loadFeats
		self._loadModels(*phones, path=self._getModelsPath(self.modelsDir, modelsSet))
		audioFeatures = loadFeats(featPath) # (numFrames, 40)
		self.scalerSet = modelsSet
		print("audioFeatures.shape", audioFeatures.shape)
		audioFeatures = self._loadScaler().transform(audioFeatures)
		
		allprobs = np.transpose([s.distribution.log_probability(audioFeatures) for m in self.models for s in m.states[:3] ])
		self._verbose("emissions shape", allprobs.shape)
		basename = os.path.basename(featPath).replace(self.ext_feat, self.ext_emissions)
		basedir = self.emissionsDir or os.path.dirname(featPath)
		savLoc = os.path.join(basedir, basename)
		if(self.ext_emissions.endswith(".txt")):
			np.savetxt(savLoc, allprobs)
		elif(self.ext_emissions.endswith(".npy")):
			np.save(savLoc, allprobs)
		else:
			with open(savLoc, "wb") as saveFile:
				pickle.dump(allprobs, saveFile)
		self._verbose(f"emissions probabilities of file {featPath} saved in {os.path.abspath(savLoc)}")

	def _loadModels(self, *args, **kwargs):
		if(not hasattr(self, "models")):
			self.models = super()._loadModels(*args, **kwargs)
	#!
	def _loadModel(self, loc):
		'''
			load the model from io in loc
		'''
		self._verbose(f"loading model from {loc}")
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

		# tick("reshaping and computing")
		features = np.array( [model.log_probability(features[int(v):int(lengths[i+1])]) for i,v  in enumerate(lengths[:-1])] )
		# tick("reshaping")
		# features = [ features[int(v):int(lengths[i+1])] for i,v  in enumerate(lengths[:-1]) ]
		# tock("done reshaping")
		# tick("compute probs")
		# tick("timing one sample")
		# model.log_probability(features[0])
		# tock("one sample")
		# features = np.array( [model.log_probability(s, check_input=False) for s in features] )
		# tock("compute probs done")
		# tock("reshaping and computing")

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
		# print(type(samples), "shape of the samples:", samples.shape)
		self._verbose("taking this sample and compute the prob of it on the model")
		logprob = model.log_probability(sample)
		print(logprob, model.probability(sample))
		return sample, path, logprob

from pomegranate.utils import is_gpu_enabled, disable_gpu
disable_gpu()
print("gpu:", is_gpu_enabled())
if __name__ == "__main__":
	from fire import Fire
	tick("timing the whole run")
	Fire(HMM_POM)
	tock("the whole run")