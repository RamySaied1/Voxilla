# this tests one file

import featuresIO as fio
from core import FeatureExtractor as FE
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf

experimentsFolder = '../Experiments'
audioTestFile='../LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac'
featuresFile=os.path.join(experimentsFolder,'feat/1272-128104-0000.feat')
shouldPlot=True

if not os.path.isfile(audioTestFile):
    raise RuntimeError('input wav file is missing. Have you downloaded the LibriSpeech corpus?')

if not os.path.exists(os.path.join(experimentsFolder,'feat')):
    os.mkdir(os.path.join(experimentsFolder,'feat'))

defaultSampleRate = 16000

speechSignal, currentSampleRate = sf.read(audioTestFile)
if (currentSampleRate != defaultSampleRate):
    raise RuntimeError("ops!!, all audio file should be 16 kHz")

fe = FE(sampleRate=defaultSampleRate, shouldNormFeature=True)


features = fe.extract(speechSignal)

if (shouldPlot):
    if not os.path.exists('fig'):
        os.mkdir('fig')

    # plot waveform
    plt.plot(speechSignal)
    plt.title('speech signal')
    plt.savefig('fig/speechSignal.png', bbox_inches='tight')
    plt.close()

    # plot mel filterbank
    fe.plotMelFilterbank(plt)
    plt.title('mel filterbank')
    plt.savefig('fig/melFilterbank.png', bbox_inches='tight')
    plt.close()

    # plot log mel spectrum (fbank)
    plt.imshow(features, origin='lower', aspect=4) # flip the image so that vertical frequency axis goes from low to high
    plt.title('log mel filterbank features (fbank)')
    plt.savefig('fig/fbank.png', bbox_inches='tight')
    plt.close()

fio.writeHtkFeatures(features, featuresFile)
print(f"Wrote {features.shape[1]} frames to {featuresFile}")
