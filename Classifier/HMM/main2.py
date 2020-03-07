import os
from my_mfcc import mfcc
# from python_speech_features import mfcc
# from pydub import AudioSegment
import numpy as np
import pickle
# from Classifier.HMM.HMMTrainer import HMMTrainer
from HMMTrainer import HMMTrainer
from align_40_phones import main as runAlign

class HMM(object):
	def __init__(self):
		self.verbose = False
		self.ext_features = ".feat.pkl"
		self.featuresDir = "models_features"
		self.ext_alignment = ".txt"

	def saveFeatures(self, *targetPhones, rootDir="data/data-generated", limit=1000, verbose=False):
		'''
			save features for *targetPhones located in rootDir. don't pass any targetPhones for saving features for all phones
			limit: pos for reading first limit lines, neg for reading last |limit| lines, 0 for reading the whole lines
		'''
		self.verbose = verbose
		phonesPaths = [os.path.join(rootDir, x) for x in os.listdir(rootDir) if x.endswith(self.ext_alignment) and (len(targetPhones) == 0 or x.replace(self.ext_alignment, "") in targetPhones)]
		for p in phonesPaths:
			samples = self._readAlignmentFile(p, limit=limit)
			features = self._getFeatures(samples)
			self._saveFeaturesForLabel(features, os.path.basename(p).replace(self.ext_alignment, ""), limit)

	def train(self, rootDir="data/data-generated", limit=1000, loadFeat=False, verbose=False):
		self.verbose = verbose
		self.trainDir = rootDir
		trainSetGenerator = self._loadFeatures(modelsSet=limit) if loadFeat else self._readTrainSet(limit=limit)
		for phoneLabel, features in trainSetGenerator:
			# input(features[0].shape, features[1])
			self._verbose("train model", phoneLabel)
			trainer = HMMTrainer()
			trainer.train(features[0], lens=features[1])
			loc = f"models/{limit}"
			os.makedirs(loc, exist_ok=True)
			loc = os.path.join(loc, phoneLabel) + ".pkl"
			with open(loc, 'wb') as saveFile: 
				pickle.dump(trainer.model, saveFile)
				self._verbose(f"model of {phoneLabel} saved in {os.path.abspath(loc)}")

	def train_phones(self, *phonesNames, rootDir="data/data-generated", limit=1000, loadFeat=False, verbose=False):
		self.verbose = verbose
		self.trainDir = rootDir
		trainSetGenerator = self._loadFeatures(*phonesNames, modelsSet=limit) if loadFeat else self._readTrainSet(limit=limit, customPhones=[os.path.basename(phoneName) + self.ext_alignment for phoneName in phonesNames] )
		for phoneLabel, features in trainSetGenerator:
			# if(saveFeat and not loadFeat): self._saveFeatures(features, phoneLabel, limit)
			self._verbose("train model", phoneLabel)
			trainer = HMMTrainer()
			trainer.train(features[0], lens=features[1])
			loc = f"models/{limit}"
			os.makedirs(loc, exist_ok=True)
			loc = os.path.join(loc, phoneLabel) + ".pkl"
			with open(loc, 'wb') as saveFile: 
				pickle.dump(trainer.model, saveFile)
				self._verbose(f"model of {phoneLabel} saved in {os.path.abspath(loc)}")
	
	def align(self, rootDir="data/librispeech_alignments-trainclean-100", outDir="data/data-generated"):
		# TODO give interface for change the target phones and phonesMapper
		self.baseAlignmentsDataDir = rootDir
		self.alignmentsOut = outDir
		runAlign(self.baseAlignmentsDataDir, self.alignmentsOut)
	
	def testScores(self, rootDir="data/data-generated", modelsDir="models/", modelsSet=200, testlimit=1, verbose=False):
		self.verbose = verbose
		self.testSetDir = rootDir
		modelsPath = os.path.join(modelsDir, str(modelsSet))
		models = self._readModels(modelsPath)
		errors = 0
		corrects = 0
		for trueLabel, features in self._readTestSet(limit=testlimit):
			scores = [trainedModel.get("model").score(features[0], lengths=features[1]) for trainedModel in models]
			bestModel = models[np.argmax(scores)].get("model")
			hypoLabel = models[np.argmax(scores)].get("label")
			max1, max2, *_ = reversed(sorted(scores))
			print("trueLabel:", trueLabel, "hypoLabel:", hypoLabel, "diff score:", abs(max1 - max2))
			if (trueLabel != hypoLabel): errors += 1
			else: corrects += 1 
		print(f"incorrect = {errors}, correct = {corrects}")

	def test_bestK(self, k=5, rootDir="data/data-generated", modelsDir="models/", modelsSet=200, testlimit=1, verbose=False):
		self.verbose = verbose
		self.testSetDir = rootDir
		modelsPath = os.path.join(modelsDir, str(modelsSet))
		models = self._readModels(modelsPath)
		errors = 0
		corrects = 0
		dists = 0
		for trueLabel, features in self._readTestSet(limit=testlimit):
			scores = [trainedModel.get("model").score(features[0], lengths=features[1]) for trainedModel in models]
			scores = np.array(scores)
			maxIndexes = (-scores).argsort()[:k]
			maxLabels = [models[i].get("label") for i in maxIndexes] # max labels is sorted
			diff = maxLabels.index(trueLabel) if trueLabel in maxLabels else None
			print("trueLabel:", trueLabel, f"first {k} hypoLabels:", maxLabels, "distance:", diff)
			if (diff != None):
				corrects += 1
				dists += diff
			else:
				errors += 1 
		print(f"incorrect = {errors}, correct = {corrects} with total distance={dists}")

	def modelsMonitor(self, modelsPath="models/", modelsSet=200):
		trainers = self._readModels(self._getModelsPath(modelsPath, modelsSet))
		for trainer in trainers:
			model, label = trainer.get("model"), trainer.get("label")
			print(label, ":", model.monitor_, "converged" if model.monitor_.converged else "not converged")
			
	def modelsInfo(self, modelsPath="models/", modelsSet=200):
		trainers = self._readModels(self._getModelsPath(modelsPath, modelsSet))
		for trainer in trainers:
			model, label = trainer.get("model"), trainer.get("label")
			print(label, ":", model.transmat_)
		
	def _verbose(self, *args, **kwargs):
		if (self.verbose):
			print(*args, **kwargs)


	# def _readAudioData(self, path, xmin=0, xmax=-1):
	# 	audio = AudioSegment.from_file(path)
	# 	selectedAudio = audio[int(xmin*1000):int(xmax*1000)]
	# 	return np.array(list(selectedAudio.raw_data)), selectedAudio.frame_rate

	def _getFeatures(self, lines):
		currentTrainFeatures = []
		lengths = []
		for sample in lines:
			tgFPath, xmin, xmax = sample
			tgFPath, xmin, xmax = tgFPath.replace("\\\\", '\\').replace("'", ""), float(xmin.replace("'", "")), float(xmax.replace("'", ""))

			audioPath = os.path.join("data\\train-clean-100", tgFPath.replace(".TextGrid", ".flac") )

			# audio, freq = self._readAudioData(audioPath, xmin=xmin, xmax=xmax)
			# features = mfcc(audio, samplerate=freq)
			features = mfcc(audioPath, start_ms=xmin*1000, stop_ms=xmax*1000)

			currentTrainFeatures.append(features)
			lengths.append(len(features))
		return np.concatenate(currentTrainFeatures), lengths

	def _readTrainSet(self, limit, customPhones=None):
		for basePhone in [x for x in os.listdir(self.trainDir) if (customPhones == None or x in customPhones)]:
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
		targetPhones = [x.replace(self.ext_features, "") for x in os.listdir(os.path.join(self.featuresDir, modelsSet)) if x.endswith(self.ext_features) and (len(targetPhones) == 0 or x.replace(self.ext_features, "") in targetPhones)]
		for label in targetPhones:
			yield label, self._loadFeaturesForLabel(label, modelsSet)

	def _readTestSet(self, limit=1000):
		for basePhone in os.listdir(self.testSetDir):
			basePhoneFPath = os.path.join(self.testSetDir, basePhone)
			currentTestFeatures = None
			with open(basePhoneFPath, 'r') as basePhoneFile:
				allLines = basePhoneFile.readlines()
				if (limit < 0):
					limit = len(allLines)
				lastLines = [x[:-1].split(" ") for x in allLines[limit:0:-1]]
				currentTestFeatures = self._getFeatures(lastLines)
			yield basePhone.replace(self.ext_alignment, ""), currentTestFeatures

	def _readModels(self, path="models/200"):
		models = []
		for phoneTrainer in [x for x in os.listdir(path) if x.endswith(".pkl")]:
			with open(os.path.join(path, phoneTrainer), "rb") as file:
				label = phoneTrainer.replace(".pkl", "")
				self._verbose("loading hmm model for", label)
				phoneHMM = pickle.load(file)
				models.append({"model": phoneHMM, "label": label})
		return models

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

	def _loadFeaturesForLabel(self, label, modelsSet):
		with open(os.path.join(self.featuresDir, str(modelsSet), label + self.ext_features), "rb") as file:
			self._verbose("loading features for", label)
			features = pickle.load(file)
		return features

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


if __name__ == "__main__":
	from fire import Fire
	Fire(HMM)