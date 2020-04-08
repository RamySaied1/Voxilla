import os
from my_mfcc import mfcc
import numpy as np
import pickle
from align_40_phones import main as runAlign
from ticktock import tick, tock
from concurrent.futures import ProcessPoolExecutor

#scaling
from sklearn.preprocessing import StandardScaler

class HMMBase(object):
	def __init__(self, 
			librispeechDir="data/train-clean-100", 
			alignmentsDir = "data/alignments",
			trainDir = "data/alignments",
			testDir = "data/alignments",
			normalizationDir = "data/normalization",
			featuresDir = "data/models-features",
			modelsDir = "data/models",
			ext_model=".model",
			n_skip=0,
			verbose=False,
			normalize=True,
			flog=False
			):
		'''
			n_skip: any sequence with number of observations less than or equal n_skip will be skipped(ignored). By default this is 0 so no ignoring. if this is 2 then any seq with 2, 1 will be ignored 
		'''
		# options
		self.verbose = verbose
		self.n_skip = n_skip
		self.normalize = normalize
		self.flog = flog # change the stdout to a file of timestamp
		# extensions
		self.ext_features = ".feat.pkl"
		self.ext_alignment = ".aligned"
		self.ext_norm = ".scale"
		self.ext_model = ext_model
		# dirs
		self.librispeechDir = librispeechDir
		self.alignmentsDir = alignmentsDir
		self.trainDir = trainDir
		self.testDir = testDir
		self.normalizationDir = normalizationDir
		self.featuresDir = featuresDir
		self.modelsDir = modelsDir

	def normalizeFeatures(self, limit=1500):
		oldNormalizeVal = self.normalize
		self.normalize = False # we should disable normalization when we are normalizing to load the intended data
		scaler = StandardScaler()
		for label, features in self._readTrainSet(limit=limit):
			self._verbose("partial scaling with features of label", label)
			scaler.partial_fit(features[0])
			self._verbose("current mean, var", scaler.mean_, scaler.var_)
			print("samples seen:", scaler.n_samples_seen_)
		loc = os.path.join(self.normalizationDir, str(limit) + self.ext_norm)
		with open(loc, "wb") as saveFile:
			pickle.dump(scaler, saveFile)
			self._verbose(f"scaler saved in {os.path.abspath(loc)}")
		self.normalize = oldNormalizeVal



	def saveFeatures(self, *targetPhones, limit=1000):
		'''
			save features for *targetPhones located in self.alignmentsDir. don't pass any targetPhones for saving features for all phones
			limit: pos for reading first limit lines, neg for reading last |limit| lines, 0 for reading the whole lines
		'''
		phonesPaths = [os.path.join(self.alignmentsDir, x) for x in sorted(os.listdir(self.alignmentsDir)) if x.endswith(self.ext_alignment) and (len(targetPhones) == 0 or x.replace(self.ext_alignment, "") in targetPhones)]
		for p in phonesPaths:
			samples = self._readAlignmentFile(p, limit=limit)
			features = self._getFeatures(samples)
			self._saveFeaturesForLabel(features, os.path.basename(p).replace(self.ext_alignment, ""), limit)

	def train(self, *phonesNames, limit=1000, loadFeat=False):
		self.scalerSet = limit
		trainSetGenerator = self._loadFeatures(*phonesNames, modelsSet=limit) if loadFeat else self._readTrainSet(limit=limit, customPhones=phonesNames)
		for phoneLabel, features in trainSetGenerator:
			# if(saveFeat and not loadFeat): self._saveFeatures(features, phoneLabel, limit)
			self._verbose("train model", phoneLabel)
			tick(f"timing total train time of {phoneLabel}")
			trainedModel = self._trainModel(phoneLabel, features)
			tock("train end")
			loc = os.path.join(self.modelsDir, str(limit))
			os.makedirs(loc, exist_ok=True)
			loc = os.path.join(loc, phoneLabel) + self.ext_model
			self._saveModel(loc, trainedModel)
			# userMessage = f"model of {phoneLabel} saved in {os.path.abspath(loc)}" if isSaved else f"model can't be saved to {loc}"
			# self._verbose(userMessage)
	def _train(self, phoneLabel, limit, loadFeat):
		self._verbose(f"{phoneLabel}: train model", phoneLabel)
		trainSetGenerator = self._loadFeatures(phoneLabel, modelsSet=limit) if loadFeat else self._readTrainSet(limit=limit, customPhones=[phoneLabel])
		label, features = next(trainSetGenerator)
		if(label != phoneLabel):
			raise RuntimeError(f"{phoneLabel}: invalid state, trainSetGenerator returns features of {label} but {phoneLabel} expected")
		tick(f"timing total train time of {phoneLabel}")
		trainedModel = self._trainModel(phoneLabel, features)
		tock(f"{phoneLabel}: train end")
		loc = os.path.join(self.modelsDir, str(limit))
		os.makedirs(loc, exist_ok=True)
		loc = os.path.join(loc, phoneLabel) + self.ext_model
		self._saveModel(loc, trainedModel)	
	def train_cores(self, *phonesNames, limit=1000, loadFeat=False):
		#from multiprocessing import cpu_count
		#from sys import __stdout__ as console
		#print(cpu_count(), file=console)
		self.scalerSet = limit
		executor = ProcessPoolExecutor(max_workers=len(phonesNames))
		for phoneLabel in phonesNames:
			print(f"scheduling {phoneLabel}")
			future = executor.submit(self._train, phoneLabel, limit, loadFeat)
			#future.result()
		executor.shutdown(True)
	
	def align(self):
		# TODO give interface for change the target phones and phonesMapper
		runAlign(self.librispeechDir, self.alignmentsDir)

	def testScores(self, *phoens, k=5, modelsSet=200, testlimit=1):
		self.scalerSet = modelsSet
		modelsPath = os.path.join(self.modelsDir, str(modelsSet))
		models = self._loadModels(*phoens, path=modelsPath)
		errors , corrects, dists = 0, 0, 0
		for trueLabel, features in self._readTestSet(limit=testlimit, rand=False):
			scores = [self._computeScore(trainedModel, features) for trainedModel in models]
			scores = np.array(scores)
			maxIndexes = (-scores).argsort()[:k]
			maxLabels = [models[i].name for i in maxIndexes] # max labels is sorted
			diff = maxLabels.index(trueLabel) if trueLabel in maxLabels else None
			self._verbose("trueLabel:", trueLabel, f"first {k} hypoLabels:", maxLabels, "distance:", diff)
			# self._verbose("maxScores", scores[maxIndexes])
			if (diff != None):
				corrects += 1
				dists += diff
			else:
				errors += 1 
		print(f"incorrect = {errors}, correct = {corrects} with total distance={dists}")
	

	def rTestScores(self, *phones, k=5, modelsSet=200, testlimit=1):
		'''
			reverse test meaning testing every model independent of other models by
			for each model m and for each different feature vectors f1,...f40 we ask m what is the prob of generating
			f1 through f40 and take maximum k scores, the corresponding label of these features vector is considered the hypoLabels
		'''
		self.scalerSet = modelsSet
		models = self._loadModels(*phones, path=self._getModelsPath(self.modelsDir, modelsSet))
		corrects, errors, totalDistance = 0, 0, 0
		for model in models:
			self._verbose(f"------model {model.name}------")
			trueLabel = model.name

			probs = [(self._computeScore(model, features), featuresLabel) for featuresLabel, features in self._readTestSet(limit=testlimit)]
			probs.sort()
			firstHypos = list(reversed(probs))[:k]
			firstHypoLabels = list(map(lambda x: x[1], firstHypos))
			self._verbose(firstHypos)
			self._verbose("true label", trueLabel, "hypoLabels", firstHypoLabels)

			if (trueLabel in firstHypoLabels):
				corrects += 1
				totalDistance += firstHypoLabels.index(trueLabel)
			else: errors += 1
		print("errors:", errors, "corrects", corrects, "with total distance =", totalDistance)

			
	def test(self, *phones, featPath=None, modelsSet=200):
		'''
			phones: restrict the phones that will be loaded. if length is 0, all available phones will be loaded
		'''
		if(featPath == None):
			raise TypeError("fpath can't be None")
		from FeatureExtraction.htk_featio import read_htk_user_feat as loadFeats
		# audioFeatures = mfcc(fpath, start_ms=0, stop_ms=None) # TODO: this is the required line (real test)
		models = self._loadModels(*phones, path=self._getModelsPath(self.modelsDir, modelsSet))
		# list(map(lambda model: input(model.name), models)) # check sorting of models
		audioFeatures = loadFeats(featPath) # (numFrames, 40)
		self.scalerSet = modelsSet
		audioFeatures = self._loadScaler().transform(audioFeatures)
		
		allprobs = np.transpose([s.distribution.log_probability(audioFeatures) for m in models for s in m.states[:3] ])
		allprobs = np.append(allprobs, allprobs[len(allprobs-3):,], axis=0)
		print("allprobs.shape", allprobs.shape)
		np.savetxt(featPath.replace(".feat", ".emis.logprob"), allprobs)


	def modelsInfo(self, *phones, modelsSet=200):
		self._verbose(f"getting information about models who trained over {modelsSet} examples")
		models = self._loadModels(*phones, path=self._getModelsPath(self.modelsDir, modelsSet))
		for model in models:
			info = self._modelInfo(model)
			print(info.name, ":\n", info.transmat)

	def generateSample(self, *phones, numSamples=1, modelsSet=200):
		models = self._loadModels(*phones, path=self._getModelsPath(self.modelsDir, modelsSet))
		for model in models:
			print(f"for model {model.name}, generating {numSamples} samples")
			sample, path, logprob = self._generateSamples(numSamples, model)
			self._verbose("samples is:", sample)
			print("paths is:", path)
			print("logprob of sample is:", logprob)

	def _verbose(self, *args, **kwargs):
		if (self.verbose):
			print(*args, **kwargs)

	def _getFeatures(self, lines, concatenate=True):
		currentTrainFeatures, lengths, ignored = [], [], 0
		for sample in lines:
			tgFPath, xmin, xmax = sample
			tgFPath, xmin, xmax = tgFPath.replace("\\\\", '/').replace("'", ""), float(xmin.replace("'", "")), float(xmax.replace("'", ""))

			audioPath = os.path.join(self.librispeechDir, tgFPath.replace(".TextGrid", ".flac") )
			features = mfcc(audioPath, start_ms=xmin*1000, stop_ms=xmax*1000)

			if (len(features) <= self.n_skip):
				ignored += 1
				continue

			currentTrainFeatures.append(features)
			lengths.append(len(features))
		if(self.n_skip > 0):
			self._verbose("_getFeatures:", "ignored", ignored, "added:", len(lengths), "ignoring%:", 100 * ignored / (ignored + len(lengths)), "%")
		if(self.normalize):
			# TODO option concatenate=False is not working when normalizing
			feats = np.concatenate(currentTrainFeatures)
			feats = self._loadScaler().transform(feats)
			return feats, lengths
		return np.concatenate(currentTrainFeatures), lengths if concatenate else currentTrainFeatures

	def _readTrainSet(self, limit, customPhones=None):
		for basePhone in [x for x in sorted(os.listdir(self.trainDir)) if (customPhones == None or len(customPhones) == 0 or x.endswith(self.ext_alignment) and x.replace(self.ext_alignment, "") in customPhones)]:
			self._verbose("reading features of phone", basePhone.replace(self.ext_alignment, ""))
			basePhoneFPath = os.path.join(self.trainDir, basePhone)
			currentTrainFeatures = None
			with open(basePhoneFPath, 'r') as basePhoneFile:
				if limit <= 0:
					allLines = basePhoneFile.readlines()
					firstLines = [line[:-1].split(" ") for line in allLines]
				else: 
					firstLines = [next(basePhoneFile)[:-1].split(" ") for x in range(limit)]
				currentTrainFeatures = self._getFeatures(firstLines)
			yield basePhone.replace(self.ext_alignment, ""), currentTrainFeatures

	def _loadFeatures(self, *targetPhones, modelsSet=1000):
		'''
			load features for *targetPhones located in self.featuresDir. don't pass any targetPhones for loading features for all phones
		'''
		modelsSet = str(modelsSet)
		refModelsSet = '0'
		targetPhones = [x.replace(self.ext_features, "") for x in os.listdir(os.path.join(self.featuresDir, refModelsSet)) if x.endswith(self.ext_features) and (len(targetPhones) == 0 or x.replace(self.ext_features, "") in targetPhones)]
		for label in targetPhones:
			yield label, self._loadFeaturesForLabel(label, modelsSet)

	def _readTestSet(self, limit=1000, rand=False):
		for basePhone in sorted(os.listdir(self.testDir)):
			basePhoneFPath = os.path.join(self.testDir, basePhone)
			currentTestFeatures = None
			with open(basePhoneFPath, 'r') as basePhoneFile:
				allLines = basePhoneFile.readlines()
				if(limit == 1 and rand):
					currentTestFeatures = self._getFeatures([allLines[np.random.randint(len(allLines))][:-1].split(" ")])
				else:
					if (limit < 0):
						limit = len(allLines)
					lastLines = [x[:-1].split(" ") for x in allLines[len(allLines) - limit:len(allLines)]]
					currentTestFeatures = self._getFeatures(lastLines)
			yield basePhone.replace(self.ext_alignment, ""), currentTestFeatures

	def _readDataSet(self, rootDir, limit=1000):
		for basePhone in sorted(os.listdir(rootDir)):
			basePhoneFPath = os.path.join(rootDir, basePhone)
			currentFeatures = None
			currentFeatures = self._getFeatures(self._readAlignmentFile(basePhoneFPath, limit=limit))
			yield basePhone.replace(self.ext_alignment, ""), currentFeatures

	def _loadModels(self, *phones, path="models/200"):
		return [self._loadModel(os.path.join(path, mPath)) for mPath in sorted(os.listdir(path)) if mPath.endswith(self.ext_model) and (len(phones) == 0 or mPath.replace(self.ext_model, "") in phones)]

	def _getModelsPath(self, modelsDir, modelsSet):
		return os.path.join(modelsDir, str(modelsSet))

	def _saveFeaturesForLabel(self, features, label, modelsSet):
		loc =  os.path.join(self.featuresDir, str(modelsSet))
		os.makedirs(loc, exist_ok=True)
		loc =  os.path.join(loc, label + self.ext_features)
		with open(loc, 'wb') as saveFile: 
			self._verbose("saving features of phone", label)
			pickle.dump(features, saveFile)
			self._verbose(f"features of {label} saved in {os.path.abspath(loc)}")

	def _loadFeaturesForLabel(self, label, modelsSet, ref=None):
		loc = os.path.join(self.featuresDir, str(modelsSet), label + self.ext_features)
		if (not os.path.exists(loc)):
			dirs = list(map(int, sorted(os.listdir(self.featuresDir))))
			dirs.sort()
			for d in dirs:
				if(d >= int(modelsSet) or d == 0):
					# higherFeatPath = os.path.join(self.featuresDir, str(d), label + self.ext_features)
					return self._loadFeaturesForLabel(label, str(d), ref=ref if ref else int(modelsSet))
			raise ValueError(f"Can't find suitable features file to read {modelsSet}, req:{ref}")
		print(f"reading features from {loc} to get {modelsSet} features and ref {ref}")
		with open(loc, "rb") as file:
			self._verbose("loading features for", label)
			features = pickle.load(file)
		features, lens = features
		features = features if ref == None or ref == 0 else features[0:ref] # filter out of limit features
		lens = lens if ref == None or ref == 0 else lens[0:ref] # filter out of limit features
		if(self.n_skip > 0): # this is added so that if there is no skip, there is no need of this operation and no need of copying
			def skipLowLens(origData, lens):
				lens = np.cumsum(lens)
				lens = np.insert(lens, 0, 0, axis=0)
				sequences = np.array([origData[int(v):int(lens[i+1])] for i,v  in enumerate(lens[:-1]) if lens[i+1] - v > self.n_skip])
				lens = list(map(len, sequences))
				return np.concatenate(sequences), lens
			features, lens = skipLowLens(features, lens)
		if (self.normalize):
			self.scalerSet = modelsSet
			self._verbose("scaling the saved features with the scaler")
			features = self._loadScaler().transform(features)
		return features, lens

	def _readAlignmentFile(self, path, limit=0):
		'''
			returns lines of the first or last limit lines in the file
		'''
		lines = []
		with open(path) as phoneAlignmentsFile:
			if (limit == 0):
				lines = [x[:-1].split(" ") for x in phoneAlignmentsFile.readlines()]
			elif (limit > 0):
				lines = [next(phoneAlignmentsFile)[:-1].split(" ") for x in range(limit)]
			else: # limit is neg
				allLines = phoneAlignmentsFile.readlines()
				lines = [x[:-1].split(" ") for x in allLines[-limit:0:-1]]
		return lines

	def _loadScaler(self):
		scalerSet = self.scalerSet
		loc = os.path.join(self.normalizationDir, str(scalerSet) + self.ext_norm)
		with open(loc, "rb") as scalerFile:
			scaler = pickle.load(scalerFile)
			return scaler
		raise RuntimeError(f"can't load the scaler file from {os.path.abspath(loc)}")

	def printFeat(self, *phones, loadFeat=False, limit=200, dataDir="data/alignments", featuresDir="data/models-features"):
		self.trainDir = dataDir
		self.featuresDir = featuresDir
		self.scalerSet = limit
		dataGenerator = self._loadFeatures(*phones, modelsSet=limit) if loadFeat else self._readTrainSet(limit=limit, customPhones=phones)
		for label, features in dataGenerator:
			print("-"*10, label, "-"*10)
			list(map(input, features[0]))

	#! these functions should be defined in the derived classes
	def _loadModel(self, loc):
		'''
			load the model from io in loc
		'''
		raise NotImplementedError("_loadModel can't be implemented in base class. you have not to directly deal with base class")

	def _saveModel(self, loc, model):
		'''
			saves the model to the given location
		'''
		raise NotImplementedError("_saveModel can't be implemented in base class. you have not to directly deal with base class")

	def _trainModel(self, label, data):
		'''
			train single model of label using data
			data is tuple of (features, lengths)
		'''
		raise NotImplementedError("_trainModel can't be implemented in base class. you have not to directly deal with base class")

	def _computeScore(self, model, data):
		'''
			computes the score of the data given the model. the max-liklihood of generating the data from this model
		'''
		raise NotImplementedError("_computeScore can't be implemented in base class. you have not to directly deal with base class")		

	def _modelInfo(self, model):
		'''
			returns the model info of the given model
		'''
		raise NotImplementedError("modelInfo can't be implemented in base class. you have not to directly deal with base class")		

	def _generateSamples(self, numSamples, model):
		raise NotImplementedError("_generateSamples can't be implemented in base class. you have not to directly deal with base class")		
		


class Selector(object):
	def __init__(self, *args, **kwargs):
		from main_hmml import HMM_HMML
		from main_pom import HMM_POM
		self.hmml = HMM_HMML(*args, **kwargs)
		self.pom = HMM_POM(*args, **kwargs)

if __name__ == "__main__":
	from fire import Fire
	tick("timing the whole run")
	Fire(HMMBase)
	tock("the whole run")