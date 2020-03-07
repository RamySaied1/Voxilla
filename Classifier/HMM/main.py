import os
from python_speech_features import mfcc
from pydub import AudioSegment
import numpy as np
import pickle
# from Classifier.HMM.HMMTrainer import HMMTrainer
from HMMTrainer import HMMTrainer
from align_40_phones import main as runAlign

class HMM(object):
	def __init__(self):
		self.verbose = False

	def train(self, rootDir="data/data-generated", limit=1000, verbose=False):
		self.verbose = verbose
		self.trainDir = rootDir
		for phoneLabel, features in self._readTrainSet(limit=limit):
			self._verbose("train model", phoneLabel)
			trainer = HMMTrainer()
			trainer.train(features)
			loc = f"models/{limit}"
			os.makedirs(loc, exist_ok=True)
			loc = os.path.join(loc, phoneLabel) + ".pkl"
			with open(loc, 'wb') as saveFile: 
				pickle.dump(trainer.model, saveFile)
				self._verbose(f"model of {phoneLabel} saved in {os.path.abspath(loc)}")

	def train_phone(self, phonePath, limit=1000, verbose=False):
		print("train_phone", phonePath, limit, verbose)
		self.verbose = verbose
		self.trainDir = os.path.dirname(phonePath)
		phoneLabel, features = next(self._readTrainSet(limit=limit, customPhones=[os.path.basename(phonePath)]))
		self._verbose("train model", phoneLabel)
		trainer = HMMTrainer()
		trainer.train(features)
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
			scores = [trainedModel.get("model").score(features) for trainedModel in models]
			bestModel = models[np.argmax(scores)].get("model")
			hypoLabel = models[np.argmax(scores)].get("label")
			max1, max2, *_ = reversed(sorted(scores))
			print("trueLabel:", trueLabel, "hypoLabel:", hypoLabel, "diff score:", abs(max1 - max2))
			if (trueLabel != hypoLabel):
				errors += 1
			else:
				corrects += 1 
				seqs = bestModel.decode(features)
		print(f"incorrect = {errors}, correct = {corrects}")

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


	def _readAudioData(self, path, xmin=0, xmax=-1):
		audio = AudioSegment.from_file(path)
		selectedAudio = audio[int(xmin*1000):int(xmax*1000)]
		return np.array(list(selectedAudio.raw_data)), selectedAudio.frame_rate

	def _getFeatures(self, lines):
		currentTrainFeatures = None
		for sample in lines:
			tgFPath, xmin, xmax = sample
			tgFPath, xmin, xmax = tgFPath.replace("\\\\", '\\').replace("'", ""), float(xmin.replace("'", "")), float(xmax.replace("'", ""))

			audioPath = os.path.join("data\\train-clean-100", tgFPath.replace(".TextGrid", ".flac") )
			audio, freq = self._readAudioData(audioPath, xmin=xmin, xmax=xmax)
			features = mfcc(audio, samplerate=freq)
			currentTrainFeatures = features if currentTrainFeatures is None else np.append(currentTrainFeatures, features, axis=0)
		return currentTrainFeatures

	def _readTrainSet(self, limit, customPhones=None):
		for basePhone in [x for x in os.listdir(self.trainDir) if (customPhones == None or x in customPhones)]:
			basePhoneFPath = os.path.join(self.trainDir, basePhone)
			currentTrainFeatures = None
			with open(basePhoneFPath, 'r') as basePhoneFile:
				firstLines = [next(basePhoneFile)[:-1].split(" ") for x in range(limit)]
				currentTrainFeatures = self._getFeatures(firstLines)
			yield basePhone.replace(".txt", ""), currentTrainFeatures

	def _readTestSet(self, limit=1000):
		for basePhone in os.listdir(self.testSetDir):
			basePhoneFPath = os.path.join(self.testSetDir, basePhone)
			currentTrainFeatures = None
			with open(basePhoneFPath, 'r') as basePhoneFile:
				allLines = basePhoneFile.readlines()
				lastLines = [x[:-1].split(" ") for x in allLines[limit:0:-1]]
				currentTrainFeatures = self._getFeatures(lastLines)
			yield basePhone.replace(".txt", ""), currentTrainFeatures

	def _readModels(self, path="models/200"):
		models = []
		for phoneTrainer in [x for x in os.listdir(path) if x.endswith(".pkl")]:
			with open(os.path.join(path, phoneTrainer), "rb") as file:
				label = phoneTrainer.replace(".pkl", "")
				self._verbose("reading hmm model for", label)
				phoneHMM = pickle.load(file)
				models.append({"model": phoneHMM, "label": label})
		return models

	def _getModelsPath(self, modelsDir, modelsSet):
		return os.path.join(modelsDir, str(modelsSet))


	def t(self):
		trainers = self._readModels()
		sampleTrainer = trainers[0]
		sampleTrainer.predict()


if __name__ == "__main__":
	from fire import Fire
	Fire(HMM)