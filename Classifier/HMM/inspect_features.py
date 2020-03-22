import os
from my_mfcc import mfcc
import numpy as np
import pickle
from ticktock import tick, tock

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from main_base import HMMBase

class Inspection(HMMBase):
	def __init__(self, librispeechDir="data/train-clean-100", inc1d=False, verbose=True, normalize=False):
		super().__init__(librispeechDir=librispeechDir, inc1d=inc1d, normalize=normalize)
		self.verbose = verbose

	def first2feat(self, *phones, limit=200):
		self.scalerSet = limit
		for phoneLabel, features in self._readTrainSet(limit=limit, customPhones=phones):
			features, seqsLens = features

			features_first2feat = np.array([observation[:2] for observation in features]) # features is of shape (n_observations, 40) we will take first twoFeat to make it (n, 2)
			self._verbose(features_first2feat.shape)
			plt.scatter(features_first2feat[:, 0], features_first2feat[:,1], label=phoneLabel)

		self._draw('feature1 of 40', 'feature2 of 40')

	def PCA(self, *phones, limit=200):
		self.scalerSet = limit
		for phoneLabel, features in self._readTrainSet(limit=limit, customPhones=phones):
			features, seqsLens = features
			reshaped = self._reshapeFeatures(features, seqsLens)

			pca = sklearnPCA(n_components=2) #2-dimensional PCA
			transformed = pd.DataFrame(pca.fit_transform(features))
			self._verbose(sum(pca.get_precision().flatten()))
			transformed = transformed.to_numpy()
			plt.scatter(transformed[:, 0], transformed[:,1], label=phoneLabel)
		self._draw("PCA1", "PCA2")
	
	def PCAoPCA(self, *phones, limit=200):
		self.scalerSet = limit
		for phoneLabel, features in self._readTrainSet(limit=limit, customPhones=phones):
			features, seqsLens = features
			reshaped = self._reshapeFeatures(features, seqsLens)

			pca = sklearnPCA(n_components=2) #2-dimensional PCA
			transformed = pd.DataFrame(pca.fit_transform(features))
			self._verbose(sum(pca.get_precision().flatten()))
			transformed = transformed.to_numpy()
			plt.scatter(transformed[:, 0], transformed[:,1], label=phoneLabel)
		self._draw("PCAoPCA1", "PCAoPCA2")

	def _draw(self, xl, yl):
		plt.xlabel(xl)
		plt.ylabel(yl)
		plt.legend()
		plt.show()


	def _reshapeFeatures(self, origData, lens):
		lens = np.cumsum(lens)
		lens = np.insert(lens, 0, 0, axis=0)
		return np.array([origData[int(v):int(lens[i+1])] for i,v  in enumerate(lens[:-1])])

def _reshapeFeatures(origData, lens):
	lens = np.cumsum(lens)
	lens = np.insert(lens, 0, 0, axis=0)
	return np.array([origData[int(v):int(lens[i+1])] for i,v  in enumerate(lens[:-1])])






def main():
	app = Inspection()
	phones = ["AY", "AA"]
	for phoneLabel, features in app._readTrainSet(limit=200, customPhones=phones):
		features, seqsLens = features
		reshaped = _reshapeFeatures(features, seqsLens)

		#! first two features only
		print("first two features only")
		# features_first2feat = np.array([observation[:2] for observation in features]) # features is of shape (n_observations, 40) we will take first twoFeat to make it (n, 2)
		# print(features_first2feat.shape)
		# plt.scatter(features_first2feat[:, 0], features_first2feat[:,1], label=phoneLabel)
		# plt.xlabel('feature1 of 40')
		# plt.ylabel('feature2 of 40')

		#! PCA
		print("PCA")
		pca = sklearnPCA(n_components=2) #2-dimensional PCA
		transformed = pd.DataFrame(pca.fit_transform(features))
		print(sum(pca.get_precision().flatten()))
		transformed = transformed.to_numpy()
		plt.scatter(transformed[:, 0], transformed[:,1], label=phoneLabel)
		plt.xlabel('PCA 1')
		plt.ylabel('PCA 2')

		#! PCA of PCA
		# print("PCA of PCA")
		# print("number of sequences:", len(seqsLens))
		# pca = sklearnPCA(n_components=2) #2-dimensional PCA 
		# print(pd.DataFrame(sklearnPCA(n_components=40).fit_transform(reshaped[0])).to_numpy())
		# input("wait")
		# pcaOfSeqs = np.array([pd.DataFrame(sklearnPCA(n_components=40).fit_transform(s)).to_numpy()[0] for s in reshaped])
		# print(pcaOfSeqs.shape)
		# transformed = pd.DataFrame(pca.fit_transform(pcaOfSeqs))
		# transformed = transformed.to_numpy()
		# plt.scatter(transformed[:, 0], transformed[:,1], label=phoneLabel)
		# plt.xlabel('PCA 1')
		# plt.ylabel('PCA 2')


		# plt.show()
	plt.legend()
	plt.show()

	# input("plotted")


if __name__ == "__main__":
	tick("timing the whole time")
	from fire import Fire
	Fire(Inspection)
	tock()

	# main()