import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import htk_featio as htk
import speech_sigproc as sp

def wav_to_feat(wav_file,samp_rate = 16000):
    x, s = sf.read(wav_file)
    if (s != samp_rate):
        raise RuntimeError("LibriSpeech files are 16000 Hz, found {0}".format(s))

    fe = sp.FrontEnd(samp_rate=samp_rate,mean_norm_feat=True)
    return fe.process_utterance(x)
