import numpy as np
import matplotlib.pyplot as plt
class FeatureExtractor:

    def __init__(self, sampleRate=16000, frameLength=0.025, frameShiftValue=0.010, preemphasis=0.97,
                 numMelFilters=40, lowFreq=0, highFreq=None, shouldNormFeature=True, shouldNormSignal=True, calcStats=False):
        self.sampleRate = sampleRate
        self.windowSize = int(np.floor(frameLength * sampleRate))
        self.windowShift = int(np.floor(frameShiftValue * sampleRate))
        self.lowFreq = lowFreq
        self.highFreq = highFreq
        if (highFreq == None):
            self.highFreq = sampleRate//2

        self.preemphasisParameter = preemphasis
        self.numMelFilters = numMelFilters
        self.fftSize = 2
        while (self.fftSize<self.windowSize):
            self.fftSize *= 2

        self.hammingWindow = np.hamming(self.windowSize)

        self.createMelFilterbank()
        self.meanNormalize = shouldNormFeature
        self.zeroMeanSignal = shouldNormSignal
        self.totalMean = np.zeros([numMelFilters])
        self.totalVariance = np.zeros([numMelFilters])
        self.totalFrames = 0
        self.calcTotalStats = calcStats
    # linear-scale frequency (Hz) to mel-scale frequency
    def lin2mel(self,freq):
        return 2595*np.log10(1+freq/700)

    # mel-scale frequency to linear-scale frequency
    def mel2lin(self,mel):
        return (10**(mel/2595)-1)*700

    def createMelFilterbank(self):

        mel_lowFreq = self.lin2mel(self.lowFreq)
        mel_highFreq = self.lin2mel(self.highFreq)

        # uniform spacing on mel scale
        melFreqs = np.linspace(mel_lowFreq, mel_highFreq,self.numMelFilters+2)

        #print("mel FREQ",melFreqs.shape)

        # convert mel freqs to hertz and then to fft bins
        binWidth = self.sampleRate/self.fftSize # typically 31.25 Hz, bin[0]=0 Hz, bin[1]=31.25 Hz,..., bin[256]=8000 Hz
        bins = np.floor(self.mel2lin(melFreqs)/binWidth)

        numBins = self.fftSize//2 + 1
        self.melFilterbank = np.zeros([self.numMelFilters,numBins])
        for i in range(self.numMelFilters):
            leftBin = int(bins[i])
            centerBin = int(bins[i+1])
            rightBin = int(bins[i+2])
            upperSlope = 1/(centerBin-leftBin)
            for j in range(leftBin,centerBin):
                self.melFilterbank[i,j] = (j - leftBin)*upperSlope
            downSlope = -1/(rightBin-centerBin)
            for j in range(centerBin,rightBin):
                self.melFilterbank[i,j] = (j-rightBin)*downSlope

        #print("mel",numBins)

    def plotMelFilterbank(self, plt):
        for i in range(self.numMelFilters):
            plt.plot(self.melFilterbank[i,:])

    ## ditehring : add random noise to the signal to solve some mahtematics issues (log (0))
    ## input : wave form (vector)
    ##output : modified wave (vector)
    def dither(self, wav):
        n = 2*np.random.rand(wav.shape[0])-1
        n *= 1/(2**15)
        return wav + n

    # pre emphizez : because important information is at high frequency so we want to boost it .  
    def preEmphasize(self, wav):
        # apply pre-emphasis filtering on waveform
        preemph_wav = []
        preemph_wav = np.append(wav[0], wav[1:] - self.preemphasisParameter * wav[:-1])
        return preemph_wav

    def cutToFrames(self, wav):
        # only process whole frames
        numFrames = int(np.floor((wav.shape[0] - self.windowSize) / self.windowShift) + 1)
        frames = np.zeros([self.windowSize, numFrames])
        for t in range(numFrames):
            frame = wav[t * self.windowShift:t * self.windowShift + self.windowSize]
            if (self.zeroMeanSignal):
                frame = frame - np.mean(frame)
            frames[:, t] = self.hammingWindow * frame
        return frames

    def getMagnitudeSpectrum(self, frames):
        magspec = np.zeros([self.fftSize//2+1,len(frames[0])])
        frames = frames.transpose()
        for i in range (0,len(frames)):
            frame_fft=np.fft.fft(frames[i])
            frame_fft=np.abs(frame_fft[0:(self.fftSize//2+1)])
            magspec[:,i]=frame_fft

        return magspec

    # for each frame(column of 2D array 'magspec'), compute the log mel spectrum, by applying the mel filterbank to the magnitude spectrum
    def calcFilterbank(self, magspec):
        fbank=self.melFilterbank.dot(magspec)
        fbank=np.log(fbank)
        return fbank

    # compute the mean vector of fbank coefficients in the utterance and subtract it from all frames of fbank coefficients
    def normalizeFilterbank(self, fbank):
        mean=np.mean(fbank,axis=1)
        fbank=fbank-mean[:,None]
        return fbank

    # accumulates sufficient statistics for corpus mean and variance
    def appendStats(self, fbank):
        self.totalMean += np.sum(fbank,axis=1)
        self.totalVariance += np.sum(fbank**2,axis=1)
        self.totalFrames += fbank.shape[1]

    # compute corpus mean and variance based on sufficient statistics
    def calcStats(self):
        self.totalMean /= self.totalFrames
        self.totalVariance /= self.totalFrames
        self.totalVariance -= self.totalMean**2

        return self.totalMean, 1.0 / np.sqrt(self.totalVariance)

    def extract(self, speechSignal):
        # preprocessing
        wav = self.dither(speechSignal)
        wav = self.preEmphasize(wav)
        frames = self.cutToFrames(wav)
        # feature calculation
        magspec = self.getMagnitudeSpectrum(frames)
        fbank = self.calcFilterbank(magspec)
        if (self.meanNormalize):
            fbank = self.normalizeFilterbank(fbank)

        if (self.calcTotalStats):
            self.appendStats(fbank)

        return fbank