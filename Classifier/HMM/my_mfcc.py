import sys
sys.path.append("../../")

import soundfile as sf
import numpy as np
from FeatureExtraction.speech_sigproc import FrontEnd as MFCCExtractor
from math import floor, ceil

def sfread(fpath, start_ms, stop_ms):
    # TODO: This is danger assumption if file is used for other dataset than librispeech
    sample_rate = 16000
    if(start_ms == None): start_ms = 0

    startFrameIndex = np.round( (start_ms / 1000) * sample_rate )
    nframes = ( (stop_ms - start_ms) / 1000) * sample_rate if stop_ms else None # (stop_ms / 1000 * sampleRate) - startFrameIndex
    stopFrameIndex = int(np.round(startFrameIndex + nframes)) if stop_ms else None
    # stopFrameIndex = ceil(startFrameIndex + nframes) #? should we use this instead
    try:
        return sf.read(fpath, start=int(startFrameIndex), stop=stopFrameIndex)
    except RuntimeError as error:
        print("Warning:", error, "startFrameIndex", startFrameIndex, "stopFrameIndex", stopFrameIndex)
        print(f"sfread({fpath}, {start_ms}, {stop_ms})") #! python 3.8 only
        res = sf.read(fpath)[0][int(startFrameIndex):int(stopFrameIndex)], sample_rate
        print("succeeded reading the whole file and access the required data only")
        return res


# startFrameIndex 238400.0 stopFrameIndex 239040.0
def mfcc(audioPath, start_ms, stop_ms):
    audio, freq = sfread(audioPath, start_ms, stop_ms)
    extractor = MFCCExtractor(samp_rate=freq, mean_norm_feat=False)
    features = extractor.process_utterance(audio)
    return np.transpose(features)

# 103\1240\103-1240-0028.TextGrid 11.67 11.7
if __name__ == "__main__":
    # sfread(fpath='data\\train-clean-100\\4441\\76263\\4441-76263-0036.flac', start_ms=14900.0, stop_ms=14940.0)
    # sfread(fpath='data\\train-clean-100\\4441\\76263\\4441-76263-0036.flac', start_ms=14900.0, stop_ms=14940.0)

    # zeros features !
    # p = 'data\\train-clean-100\\103\\1240\\103-1240-0028.flac'
    # s, e = 11.67*1000, 11.7*1000
    p = 'data\\train-clean-100\\1081\\125237\\1081-125237-0024.flac'
    s, e = 11.47*1000, 11.5*1000
    audio, sampleRate = sfread(fpath=p, start_ms=s, stop_ms=e)
    print(audio.shape)
    print(mfcc(p, s, e))
    sf.write("zerosFeatures.flac", audio, sampleRate)
